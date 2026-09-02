from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Examples.Demo.start_pysc_demo_server import build_machine, catalog_for, configure_demo_bpm_noise, install_server_compatibility_shim


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
