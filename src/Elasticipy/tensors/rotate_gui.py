from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QPushButton
)
from qtpy.QtCore import Qt, Signal


class EulerBungeDialog(QDialog):
    anglesChanged = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Euler angles (Bunge ZXZ)")
        self._build_ui()
        self.reset()

    def _build_ui(self):
        self.layout = QVBoxLayout(self)

        self.sliders = []
        self.spins = []

        labels = ["φ₁ (deg)", "Φ (deg)", "φ₂ (deg)"]
        ranges = [(0, 360), (0, 180), (0, 360)]

        for label, (vmin, vmax) in zip(labels, ranges):
            row = QHBoxLayout()

            row.addWidget(QLabel(label))

            slider = QSlider(Qt.Horizontal)
            slider.setRange(vmin * 10, vmax * 10)
            slider.setSingleStep(1)

            spin = QDoubleSpinBox()
            spin.setRange(vmin, vmax)
            spin.setDecimals(1)
            spin.setSingleStep(0.1)

            slider.valueChanged.connect(
                lambda v, s=spin: s.setValue(v / 10)
            )
            spin.valueChanged.connect(
                lambda v, s=slider: s.setValue(int(v * 10))
            )
            spin.valueChanged.connect(self._emit_angles)

            row.addWidget(slider, stretch=1)
            row.addWidget(spin)

            self.sliders.append(slider)
            self.spins.append(spin)

            self.layout.addLayout(row)

        # Reset button
        reset_button = QPushButton("Reset orientation")
        reset_button.clicked.connect(self.reset)
        self.layout.addWidget(reset_button)

    def reset(self):
        self._set_angles(0.0, 0.0, 0.0)

    def _set_angles(self, phi1, Phi, phi2):
        for spin, val in zip(self.spins, (phi1, Phi, phi2)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._emit_angles()

    def _emit_angles(self):
        angles = [spin.value() for spin in self.spins]
        self.anglesChanged.emit(*angles)
