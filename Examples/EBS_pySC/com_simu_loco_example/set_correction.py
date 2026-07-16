import numpy as np

def last_by_sorted_key(d):
    def norm(k):
        # try numeric sort first; fallback to string
        try:
            return (0, float(k))
        except Exception:
            return (1, str(k))
    last_key = sorted(d.keys(), key=norm)[-1]
    return d[last_key]

def get_quads_block(fit_vec):
    """
    Extract (quads_fit, skew_fit) from the *last* entry of fit_vec.

    Accepted last-entry formats:
      - dict with keys 'quads' and optionally 'skew' or 'skew_quads'
      - tuple/list like (quads, skew)
      - plain ndarray/sequence of quads only
    Returns:
      (np.ndarray, np.ndarray)  # skew may be empty if not available
    Missing values -> np.nan.
    """
    # empty / None guard
    if fit_vec is None:
        return np.asarray(np.nan), np.asarray(np.nan)
    try:
        last = fit_vec[-1]  # works for list/tuple/ndarray
    except Exception:
        # fit_vec is scalar or not indexable
        last = fit_vec

    # Case 1: dict-like
    if isinstance(last, dict):
        quads = np.asarray(last.get('quads', np.nan), dtype=float)
        # accept several possible names for skew block
        skew = last.get('skew', last.get('skew_quads', np.nan))
        skew = np.asarray(skew, dtype=float)
        return quads, skew

    # Case 2: tuple/list: (quads, skew) or just [quads]
    if isinstance(last, (list, tuple)):
        if len(last) == 0:
            return np.asarray(np.nan), np.asarray(np.nan)
        if len(last) >= 2:
            quads = np.asarray(last[0], dtype=float)
            skew  = np.asarray(last[1], dtype=float)
            return quads, skew
        # single-entry -> treat as quads only
        return np.asarray(last[0], dtype=float), np.asarray([])

    # Case 3: NumPy array
    if isinstance(last, np.ndarray):
        # If it's a structured array with named fields, try fields first
        if last.dtype.names:
            q = last[last.dtype.names[0]] if 'quads' not in last.dtype.names else last['quads']
            quads = np.asarray(q, dtype=float)
            if 'skew' in last.dtype.names:
                skew = np.asarray(last['skew'], dtype=float)
            elif 'skew_quads' in last.dtype.names:
                skew = np.asarray(last['skew_quads'], dtype=float)
            else:
                skew = np.asarray([])
            return quads, skew
        # plain numeric array -> assume it's quads only
        return np.asarray(last, dtype=float), np.asarray([])

    # Fallback: unknown type -> return NaNs
    return np.asarray(np.nan), np.asarray(np.nan)




def set_correction_(SC, r, elem_ind, skewness=False):
    if skewness:
       comp = "/A2"
    else:
       comp = "/B2"
    name = SC.magnet_settings.index_mapping[elem_ind]+comp

    data = SC.magnet_settings.get(name)
    SC.magnet_settings.set(name, data+r)



def set_correction(SC, r, elem_ind, individuals=True, skewness=False):

    if len(r) != len(elem_ind):
        raise ValueError(f"Length mismatch: len(r)={len(r)}, len(elem_ind)={len(elem_ind)}")

    if individuals == True:

        for i, quad_idx in enumerate(elem_ind):
            fit_value = r[i]

            set_correction_(SC, fit_value, quad_idx,
                                skewness=skewness)

    else:
        for fam_num, quad_fam in enumerate(elem_ind):

            r2 = r[fam_num]
            for i, quad_idx in enumerate(quad_fam):

                fit_value = r2

                set_correction_(SC, fit_value, quad_idx,
                                    skewness=skewness)




def set_correction0(SC, r, elem_ind, individuals=True, skewness=False, order=1, method='add',
                   dipole_compensation=False):

    from collections import defaultdict

    if np.isscalar(elem_ind):
        elem_ind = [int(elem_ind)]
    else:
        elem_ind = list(elem_ind)

    if np.isscalar(r):
        r = np.full(len(elem_ind), r)
    else:
        r = np.array(r)

    if len(r) != len(elem_ind):
        raise ValueError(f"Length mismatch: len(r)={len(r)}, len(elem_ind)={len(elem_ind)}")



    for i, quad_idx in enumerate(elem_ind):

        fit_value = r[i]
        set_correction_(SC, fit_value, quad_idx, individuals=True,
                            skewness=skewness, order=order, method=method,
                            dipole_compensation=dipole_compensation)

    return SC

