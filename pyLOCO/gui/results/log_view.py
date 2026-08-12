"""Backend execution log view."""

from PySide6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QTextEdit, QVBoxLayout, QWidget


class LogView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self.text.toPlainText()))
        clear_button = QPushButton("Clear display")
        clear_button.clicked.connect(self.text.clear)
        row = QHBoxLayout(); row.addWidget(copy_button); row.addWidget(clear_button); row.addStretch(1)
        layout = QVBoxLayout(self); layout.addLayout(row); layout.addWidget(self.text, 1)

    def append(self, message: str) -> None:
        self.text.append(message)

    def set_log(self, text: str) -> None:
        self.text.setPlainText(text)

    def clear_for_run(self) -> None:
        self.text.clear()

