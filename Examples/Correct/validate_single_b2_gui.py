"""Native GUI validation against the isolated PETRA simulation (one B2 only)."""
import json
import argparse
from pathlib import Path
from PySide6.QtCore import QTimer
from pyLOCO.correct.app import build_application
from pyLOCO.correct.main_window import CorrectMainWindow

app = build_application(['correct-single-b2-validation'])
parser = argparse.ArgumentParser(); parser.add_argument('--request', default='Examples/Correct/petra_single_b2.json')
args = parser.parse_args()
window = CorrectMainWindow(); window.resize(1200, 800)
window.backend_combo.setCurrentIndex(window.backend_combo.findData('pysc'))
w = window.quadrupole_workspace
w.profile.setCurrentIndex(w.profile.findData('petra3_realistic')); w.port.setValue(13332)
window.tabs.setCurrentIndex(4); window.show()

def preview():
    w.connect_profile()
    w.request = json.loads(Path(args.request).read_text())
    w.preview()
    window.grab().save('/private/tmp/correct-single-b2-preview.png')

QTimer.singleShot(1000, preview)
app.exec()
