"""GET-only native-window validation of a real FIT bundle; never applies."""
import argparse
from PySide6.QtCore import QTimer
from pyLOCO.correct.app import build_application
from pyLOCO.correct.main_window import CorrectMainWindow

parser = argparse.ArgumentParser()
parser.add_argument('--bundle', default='Examples/Correct/petra_noerr16_fit_preview.json')
parser.add_argument('--diagnostics-port', type=int, default=13332)
parser.add_argument('--screenshot', default='/private/tmp/correct-real-fit-preview.png')
args = parser.parse_args()
app = build_application(['validate-real-fit-preview'])
window = CorrectMainWindow(); window.setWindowTitle('pyLOCO Correct — Real FIT preview only')
window.resize(1450, 950)
widget = window.quadrupole_workspace
window.backend_combo.setCurrentIndex(window.backend_combo.findData('pysc'))
widget.profile.setCurrentIndex(widget.profile.findData('petra3_realistic'))
widget.port.setValue(args.diagnostics_port)
window.tabs.setCurrentIndex(4); window.show()


def validate():
    widget.connect_profile()
    connection = widget.transaction.connection
    original = connection.snapshot()
    interface_factory = connection.interface
    reads = []
    class ReadOnly:
        def __init__(self, identity): self.delegate = interface_factory(identity)
        def get(self, name):
            reads.append(name)
            return self.delegate.get(name)
        def set(self, *args): raise AssertionError('FIT preview must never write')
    connection.interface = ReadOnly
    widget.load_fit_path(args.bundle); widget.preview()
    assert original == connection.snapshot()
    assert not widget.apply_button.isEnabled()
    assert len(reads) == widget.fit_record['mapped'] == 398
    print(widget.summary.text(), flush=True)
    print(f'Control GETs: {len(reads)}; SETs: 0; machine unchanged; Apply disabled', flush=True)
    QTimer.singleShot(1000, lambda: window.grab().save(args.screenshot))
    QTimer.singleShot(15000, app.quit)


QTimer.singleShot(100, validate)
app.exec()
