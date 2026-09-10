"""Audited individual-normal-K FIT bundles. Deliberately no write API.

The manifest supplies provenance, not correction numbers. K is read directly
from the real initial/fitted lattices; source ordinals never address the server.
"""
import hashlib
import json
import math
from pathlib import Path

from .quadrupole_transaction import close


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def element_key(element):
    key = (str(getattr(element, 'CommonName', '')), str(element.FamName))
    if not all(key):
        raise ValueError('Explicit CommonName AND FamName required')
    return key


def load_fit_bundle(path, profile_key='petra3_realistic'):
    """Load a hash-pinned legacy FIT result without unpickling fit dictionaries."""
    import at
    import numpy as np
    from pyLOCO.control_system.pysc_profiles import load_pysc_profile

    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'pyloco-fit-b2-preview-v1' or
            manifest.get('parameter_mode') != 'individual' or
            manifest.get('unit') != 'm^-2'):
        raise ValueError('Only audited individual normal K in m^-2 is supported; no family expansion')
    sources = {}
    for name in ('initial', 'fitted', 'ordinals', 'provenance'):
        entry = manifest['sources'][name]
        source = (path.parent / entry['path']).resolve()
        if digest(source) != entry['sha256']:
            raise ValueError(f'{name} source hash mismatch')
        sources[name] = str(source)
    official = load_pysc_profile(profile_key).resolve('lattice_file')
    if digest(official) != manifest['sources']['initial']['sha256']:
        raise ValueError(f'FIT initial lattice is not the official {profile_key} baseline')
    initial = at.load_lattice(sources['initial'])
    fitted = at.load_lattice(sources['fitted'])
    if len(initial) != len(fitted) or initial.energy != fitted.energy:
        raise ValueError('Initial/fitted lattice structure or energy mismatch')
    if any(element_key(a) != element_key(b) or a.Length != b.Length
           for a, b in zip(initial, fitted)):
        raise ValueError('Initial/fitted lattice element identities differ')
    ordinals = np.load(sources['ordinals'], allow_pickle=False)
    if (ordinals.ndim != 1 or not np.issubdtype(ordinals.dtype, np.integer) or
            len(ordinals) == 0 or len(set(ordinals.tolist())) != len(ordinals) or
            np.any(ordinals < 0) or np.any(ordinals >= len(initial))):
        raise ValueError('Invalid or duplicate FIT source ordinals')
    parameters = []
    for ordinal in ordinals:
        a, b = initial[int(ordinal)], fitted[int(ordinal)]
        k0, k1 = float(a.PolynomB[1]), float(b.PolynomB[1])
        if not all(math.isfinite(k) for k in (k0, k1)):
            raise ValueError('Nonfinite FIT K')
        parameters.append(dict(source_ordinal=int(ordinal), common_name=element_key(a)[0],
                               family=element_key(a)[1], initial=k0, fitted=k1,
                               full_delta=k0-k1))
    keys = [(r['common_name'], r['family']) for r in parameters]
    if len(set(keys)) != len(keys):
        raise ValueError('Ambiguous FIT element identity')
    return dict(parameters=parameters, sources=sources, manifest=manifest, profile=profile_key,
                lattice_sha256=digest(official), fraction=0.1, unit='m^-2',
                convention='recommended physical delta K = initial K - fitted K')


def preview_fit(bundle, connection):
    """GET-only preview; unrelated historical fits are never auto-applied."""
    snapshot = connection.snapshot()
    identity = snapshot['identity']
    expected_profile = bundle.get('profile', 'petra3_realistic')
    if (identity.get('profile') != expected_profile or
            identity['lattice_sha256'] != bundle['lattice_sha256']):
        raise ValueError(f'Requires matching {expected_profile} official lattice')
    inventory = {}
    for row in snapshot['quadrupoles']:
        inventory.setdefault((row['common_name'], row['family']), []).append(row)
    mappings, unmapped, ambiguous = [], [], []
    for parameter in bundle['parameters']:
        key = (parameter['common_name'], parameter['family'])
        candidates = inventory.get(key, [])
        if not candidates:
            unmapped.append(key)
        elif len(candidates) != 1:
            ambiguous.append(key)
        else:
            mappings.append((parameter, candidates[0]))
    if unmapped or ambiguous:
        raise ValueError(f'Incomplete mapping: {len(mappings)} mapped; unmapped={unmapped}; ambiguous={ambiguous}')
    if len({r['control'] for _, r in mappings}) != len(mappings):
        raise ValueError('Multiple FIT parameters map to one control')
    interface = connection.interface(identity)
    items = []
    for parameter, row in mappings:
        if row['component'] != 'B2' or row['unit'] != 'm^-2':
            raise ValueError('Only B2 / m^-2 allowed')
        if not all(math.isfinite(row[k]) for k in ('factor', 'offset', 'current', 'physical')) or row['factor'] == 0:
            raise ValueError('Invalid simulation calibration')
        if not close(interface.get(row['control']), row['current']):
            raise ValueError('Machine changed during preview; preview again')
        if not close(row['physical'], row['factor'] * row['current'] + row['offset']):
            raise ValueError('Unsupported physical/control calibration')
        delta = 0.1 * parameter['full_delta']
        items.append(dict(parameter=parameter, original=row, applied_delta=delta,
                          control_delta=delta/row['factor'], proposed=row['current']+delta/row['factor'],
                          expected_physical=row['physical']+delta))
    after = connection.snapshot()
    if after != snapshot:
        raise ValueError('Server state changed during preview; preview again')
    deltas = [p['full_delta'] for p in bundle['parameters']]
    return dict(status='FIT PREVIEW ONLY — Apply disabled', identity=identity, items=items,
                source=bundle, mapped=len(items), unmapped=0, ambiguous=0,
                rms_delta=math.sqrt(sum(d*d for d in deltas)/len(deltas)),
                max_delta=max(abs(d) for d in deltas))
