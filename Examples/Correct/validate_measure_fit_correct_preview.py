"""Visible reduced Measure acquisition for the fixed-seed FIT/Correct proof."""
from pathlib import Path
import json, sys, time
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from pyLOCO.measure.app import build_application
from pyLOCO.measure.main_window import MeasureMainWindow
from pyLOCO.control_system.backends import InterfaceRegistry
from pyLOCO.control_system.pysc_server import pySCServerOrbitInterface

app=build_application(['measure-fit-correct-proof'])
for name in ('information','warning','critical'):
 setattr(QMessageBox,name,staticmethod(lambda *a,**k: (print('DIALOG',a[1:3],flush=True),QMessageBox.Ok)[1]))
w=MeasureMainWindow(); w.resize(1400,900); w.output_directory.setText('/private/tmp/pyloco-measure-fit-proof')
w.adapter_combo.setCurrentIndex(w.adapter_combo.findData('pysc')); w.pysc_profile_combo.setCurrentIndex(w.pysc_profile_combo.findData('petra3_realistic'))
session=InterfaceRegistry(interface_loaders={'pysc':lambda:pySCServerOrbitInterface(host='127.0.0.1',port=13331)},pysc_profile='petra3_realistic').create('pysc')
w.adapter=session.adapter
w.show(); state={'phase':'connect','files':{}}; started=time.monotonic()
def begin(kind):
 print('BEGIN',kind,flush=True)
 w.measurement_type.setCurrentIndex(w.measurement_type.findData(kind)); w._select_device_subset('bpm','uniform',40)
 if kind=='orm': w._select_device_subset('hcor','uniform',5);w._select_device_subset('vcor','uniform',5);w.orm_scaled.setChecked(False)
 w.readings.setValue(5);w.delay.setValue(0);w.settling_delay.setValue(0)
 if kind=='dispersion': w.rf_step.setValue(3000);w.verify_restored_orbit.setChecked(True)
 w.measurement_name.setText('same-machine-'+kind);w.measurement_label.setText('PETRA realistic seed 20260907 '+kind);w.refresh_preview();w.start_button.click();state['phase']=kind
def tick():
 if time.monotonic()-started>240: print('TIMEOUT',state,flush=True);app.exit(2);return
 if state['phase']=='connect':
  result=w.adapter.test_connection();w._set_connection_state(True,'CONNECTED');w.nominal_rf.setText(f"{result['rf_readback']:.12f}")
  print('CONNECTED',result,flush=True)
  begin('bpm_noise')
 elif state['phase'] in ('bpm_noise','dispersion','orm') and w.thread is None and w.result is not None:
  kind=state['phase'];state['files'][kind]=str(w.saved_measurement_path);print(kind,state['files'][kind],flush=True)
  if kind=='bpm_noise':begin('dispersion')
  elif kind=='dispersion':begin('orm')
  else:
   Path('/private/tmp/pyloco-measure-fit-proof/measure-files.json').write_text(json.dumps(state['files'],indent=2));w.grab().save('/private/tmp/measure-fit-proof.png');w.close();app.exit(0);return
 QTimer.singleShot(100,tick)
QTimer.singleShot(200,tick);raise SystemExit(app.exec())
