from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Examples.Demo.start_pysc_demo_server import build_machine, catalog_for, configure_demo_bpm_noise, install_server_compatibility_shim
from Examples.Demo.start_pysc_server import (build_profile_machine, catalog_for as profile_catalog_for,
                                              realized_error_statistics)
from pyLOCO.control_system.pysc_profiles import load_pysc_profile


def test_demo_noise_override_is_local_and_configurable():
    bpm=SimpleNamespace(names=("a","b","c"),noise_co_x=np.zeros(3),noise_co_y=np.zeros(3))
    configure_demo_bpm_noise(SimpleNamespace(bpm_system=bpm),sigma_x_m=1.5e-6,sigma_y_m=2e-6)
    np.testing.assert_array_equal(bpm.noise_co_x,np.full(3,1.5e-6))
    np.testing.assert_array_equal(bpm.noise_co_y,np.full(3,2e-6))


def test_demo_catalog_records_at_and_accelerator_slip_factor_conventions():
    sc=build_machine(); ring=sc.lattice.ring.disable_6d(copy=True); metadata=catalog_for(sc,host="127.0.0.1",port=13131)["metadata"]
    assert metadata["slip_factor"]==ring.get_slip_factor()
    assert metadata["eta_alpha_minus_inverse_gamma_squared"]==-ring.get_slip_factor()
    np.testing.assert_allclose(
        metadata["momentum_compaction_factor"]-metadata["relativistic_correction_inverse_gamma_squared"],
        -ring.get_slip_factor(),rtol=1e-12,atol=0,
    )


def test_official_petra_profile_builds_authoritative_inventory_and_rf():
    profile,sc=build_profile_machine("petra3")
    catalog=profile_catalog_for(sc,profile,host="127.0.0.1",port=13131)
    assert profile.configuration["provenance"]["lattice_sha1"]=="2c966698981886f172ece1bffc33eaaa4e13adfc"
    assert len(catalog["bpms"])==246
    assert len(catalog["horizontal_correctors"])==219
    assert len(catalog["vertical_correctors"])==194
    assert catalog["bpms"]==list(sc.bpm_system.names)
    assert catalog["horizontal_correctors"]==list(sc.tuning.HCORR)
    assert catalog["vertical_correctors"]==list(sc.tuning.VCORR)
    assert sc.rf_settings.systems["main"].frequency==pytest.approx(499_664_399.4230214)
    assert np.all(sc.bpm_system.noise_co_x==15e-6)
    assert np.all(sc.bpm_system.noise_co_y==15e-6)


def test_profile_paths_keep_official_petra_lattice_separate_from_live_data():
    profile=load_pysc_profile("petra3")
    assert profile.resolve("lattice_file").is_relative_to(profile.directory)
    assert profile.resolve("lattice_file") != (Path(__file__).parents[1]/"Examples/PETRAIII/data/p3_v24.mat").resolve()


def test_realistic_petra_profile_is_fixed_seed_uncorrected_and_keeps_inventory():
    profile,sc=build_profile_machine("petra3_realistic")
    catalog=profile_catalog_for(sc,profile,host="127.0.0.1",port=13131)
    assert profile.configuration["random_seed"]==20260907
    assert profile.configuration["sigma_truncate"]==2.5
    assert profile.configuration["uncorrected"] is True
    assert (len(catalog["bpms"]),len(catalog["horizontal_correctors"]),len(catalog["vertical_correctors"]))==(246,219,194)
    realized=realized_error_statistics(sc)
    assert realized["quadrupole_relative_calibration"]["count"]==417
    assert realized["bpm_relative_gain_x"]["rms"]==pytest.approx(0.0047880209312772615)
    assert realized["hcor_relative_calibration"]["rms"]==pytest.approx(0.004628030460374097)
    orbit=np.asarray(sc.lattice.get_orbit(indices=sc.bpm_system.indices))
    assert np.isfinite(orbit).all()
    assert np.sqrt(np.mean(orbit[0]**2))==pytest.approx(2.154373917029668e-05)
    assert np.sqrt(np.mean(orbit[1]**2))==pytest.approx(4.8179961559650683e-05)
    rf=sc.rf_settings.main
    assert len(rf.cavities)==len(set(rf.cavities))==12
    assert len(rf.indices)==len(set(rf.indices))==12
    original=rf.frequency
    try:
        for frequency in (original+1500.0,original-1500.0,original):
            rf.set_frequency(frequency)
            assert sc.lattice.ring.get_rf_frequency()==pytest.approx(frequency,abs=0)
            cavity_frequencies=[
                sc.lattice.get_cavity_voltage_phase_frequency(index,use_design=False)[2]
                for index in rf.indices
            ]
            np.testing.assert_array_equal(cavity_frequencies,np.full(12,frequency))
    finally:
        rf.set_frequency(original)


def test_orbit_requests_are_fresh_and_xy_paired(monkeypatch):
    import pySC.control_system.send_receive as send_receive
    import pySC.control_system.server as server

    sent = []
    monkeypatch.setattr(send_receive, "send_int", lambda conn, value: None)
    monkeypatch.setattr(
        send_receive,
        "send_nparray",
        lambda conn, value: sent.append(np.asarray(value).copy()),
    )

    captures = []

    def capture_orbit():
        number = len(captures) + 1
        pair = (np.array([number, 10 + number]), np.array([-number, -10 - number]))
        captures.append(pair)
        return pair

    sc = SimpleNamespace(
        bpm_system=SimpleNamespace(capture_orbit=capture_orbit),
        tuning=SimpleNamespace(correct_injection=lambda parameter: None),
    )

    install_server_compatibility_shim()
    handler = server.orbit_server
    cached = np.array([999.0])
    handler(None, "GET ORBIT/RAW/X", cached, cached, sc)
    handler(None, "GET ORBIT/RAW/Y", cached, cached, sc)
    handler(None, "GET ORBIT/RAW/X", cached, cached, sc)
    handler(None, "GET ORBIT/RAW/Y", cached, cached, sc)

    assert len(captures) == 2
    np.testing.assert_array_equal(sent[0], captures[0][0])
    np.testing.assert_array_equal(sent[1], captures[0][1])
    np.testing.assert_array_equal(sent[2], captures[1][0])
    np.testing.assert_array_equal(sent[3], captures[1][1])


def test_pysc_bpm_noise_protocol_is_orbit_reads_only(monkeypatch):
    """A complete BPM-noise run must never issue RF or magnet SET commands."""
    import pyLOCO.control_system.pysc_server as protocol
    from pyLOCO.control_system.backends import AbstractInterfaceAdapter
    from pyLOCO.measure.acquisition import BpmDevice, BpmNoiseAcquirer

    counts={"orbit_reads":0,"captures":0,"rf_sets":0,"magnet_sets":0}
    pending={}

    def fake_read(address):
        variable=address.split("/",1)[1]
        if variable=="ORBIT/RAW/X":
            counts["captures"]+=1; number=float(counts["captures"])
            pending["pair"]=(np.array([number,number+1]),np.array([-number,-number-1]))
            counts["orbit_reads"]+=1; return pending["pair"][0]
        if variable=="ORBIT/RAW/Y":
            counts["orbit_reads"]+=1; return pending.pop("pair")[1]
        raise AssertionError(f"Unexpected BPM-noise GET: {variable}")

    def fake_write(address,value):
        variable=address.split("/",1)[1]
        if variable.startswith("RF/"):counts["rf_sets"]+=1
        elif variable.startswith("MAGNET/"):counts["magnet_sets"]+=1
        else:raise AssertionError(f"Unexpected BPM-noise SET: {variable}")

    monkeypatch.setattr(protocol,"read",fake_read)
    monkeypatch.setattr(protocol,"write",fake_write)
    interface=protocol.pySCServerOrbitInterface(host="instrumented",port=13131)
    adapter=AbstractInterfaceAdapter(interface,("BPM-1","BPM-2"),(),())
    devices=tuple(BpmDevice(**item) for item in adapter.list_devices("bpm"))
    delays=[]
    result=BpmNoiseAcquirer(adapter,devices).acquire(20,0.1,sleeper=delays.append)

    assert result.orbits_x_m.shape==(20,2)
    assert counts["captures"]==20
    assert counts["orbit_reads"]==40
    assert counts["rf_sets"]==0
    assert counts["magnet_sets"]==0
    assert delays==[0.1]*19
