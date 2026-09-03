"""Read-only startup probe; run each case in a fresh process.

QT_QPA_PLATFORM=offscreen .venv/bin/python Examples/Correct/profile_startup.py correct
Use suite_correct for the actual FIT open_correct_app handler (no results loaded).
Omit QT_QPA_PLATFORM for native macOS window timings. No machine is contacted.
"""
import importlib
import json
import sys
from time import perf_counter

kind = sys.argv[1]
times = {}
origin = perf_counter()
def mark(name):
    times[name] = round(1000 * (perf_counter() - origin), 3)

if kind == "suite_correct":
    fit = importlib.import_module("pyLOCO.gui.app")
    app = fit.build_application(["startup-probe"])
    owner = fit.MainWindow()
    owner.show()
    app.processEvents()
    origin = perf_counter()
    mark("click_correct")
    module = importlib.import_module("pyLOCO.correct.main_window")
    cls = module.CorrectMainWindow
else:
    module = importlib.import_module({"correct":"pyLOCO.correct.app", "measure":"pyLOCO.measure.app", "fit":"pyLOCO.gui.app"}[kind])
    cls = getattr(module, {"correct":"CorrectMainWindow", "measure":"MeasureMainWindow", "fit":"MainWindow"}[kind])
mark("module_import_done")
if kind != "suite_correct":
    app = module.build_application(["startup-probe"])
mark("qapplication_ready")
if kind in {"correct", "suite_correct"}:
    original_init = cls.__init__
    def initialize(self, *args, **kwargs):
        mark("window_construction_start")
        original_init(self, *args, **kwargs)
        mark("window_construction_done")
    cls.__init__ = initialize
    original_show = cls.show
    def show(self):
        result = original_show(self)
        if "first_show_returned" not in times:mark("first_show_returned")
        return result
    cls.show = show
    registry = sys.modules["pyLOCO.correct.main_window"].InterfaceRegistry
    original_registry = registry.__init__
    def initialize_registry(self, *args, **kwargs):
        start = perf_counter()
        original_registry(self, *args, **kwargs)
        times["backend_registry_duration_ms"] = round(1000*(perf_counter()-start),3)
    registry.__init__ = initialize_registry
    original_build = cls._build
    def build(self):
        mark("model_registry_ready_ui_start")
        original_build(self)
        mark("ui_done")
    cls._build = build
    for name in ("_source_page", "_mapping_page", "_plan_page", "_review_page"):
        original = getattr(cls, name)
        def page(self, original=original, name=name):
            start = perf_counter()
            result = original(self)
            times[name + "_duration_ms"] = round(1000 * (perf_counter()-start), 3)
            return result
        setattr(cls, name, page)
if kind == "suite_correct":
    owner.open_correct_app()
    window = owner._correct_window
else:
    window = cls()
mark("window_constructed")
window.show()
mark("show_returned")
from PySide6.QtCore import QTimer
def ready():
    mark("first_event_loop_callback")
    print(json.dumps({"kind":kind,"timings_ms":times,"theme":app.property("pyLOCOTheme"),"empty":getattr(window,"review",None) is None,"loaded_modules":{name:name in sys.modules for name in ("at","scipy","matplotlib","pyLOCO.pyloco")}}, sort_keys=True), flush=True)
    window.close()
    if kind == "suite_correct":owner.close()
    app.quit()
QTimer.singleShot(0, ready)
app.exec()
