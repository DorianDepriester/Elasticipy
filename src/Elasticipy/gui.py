import sys
import numpy as np
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QGridLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QMessageBox
)
from qtpy.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Elasticipy.crystal_symmetries import SYMMETRIES
from Elasticipy.tensors.elasticity import StiffnessTensor
from Elasticipy.tensors.fourth_order import SymmetricFourthOrderTensor
from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from pathlib import Path

from scipy.spatial.transform import Rotation

from Elasticipy.tensors.mapping import VoigtMapping

WHICH_OPTIONS = {'Mean': 'mean', 'Max': 'max', 'Min': 'min', 'Std. dev.': 'std'}

class ElasticityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.coefficient_fields = {}
        self.setWindowTitle("Elasticipy - GUI")
        self.initUI()

    def selected_symmetry(self):
        symmetry_name = self.symmetry_selector.currentText()
        symmetry = SYMMETRIES[symmetry_name]
        if symmetry_name == "Trigonal" or symmetry_name == "Tetragonal":
            space_group_text = self.point_group_selector.currentText()
            try:
                symmetry = symmetry[space_group_text]
            except KeyError:
                # If no SG is selected, just choose one
                symmetry = list(symmetry.values())[0]
        elif symmetry_name == "Monoclinic":
            diad_index = self.diag_selector.currentText()
            symmetry = symmetry[diad_index]
        return symmetry

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()

        #######################################################################################
        # Material's symmetry and other parameters
        #######################################################################################
        selectors_layout = QHBoxLayout()

        # Symmetry selection
        self.symmetry_selector = QComboBox()
        self.symmetry_selector.addItems(SYMMETRIES.keys())
        self.symmetry_selector.currentIndexChanged.connect(self.update_fields)
        selectors_layout.addWidget(QLabel("Crystal symmetry:"))
        selectors_layout.addWidget(self.symmetry_selector)

        # Space Group selection
        self.point_group_selector = QComboBox()
        self.point_group_selector.setMinimumWidth(140)
        self.point_group_selector.addItems(['', ''])
        self.point_group_selector.currentIndexChanged.connect(self.update_fields)
        selectors_layout.addWidget(QLabel("Point group:"))
        selectors_layout.addWidget(self.point_group_selector)

        # Diad selection
        self.diag_selector = QComboBox()
        self.diag_selector.addItems(SYMMETRIES['Monoclinic'].keys())
        self.diag_selector.currentIndexChanged.connect(self.update_fields)
        selectors_layout.addWidget(QLabel("Diad convention:"))
        selectors_layout.addWidget(self.diag_selector)

        # About button
        self.about_button = QPushButton("About")
        self.about_button.setFixedHeight(24)
        self.about_button.setMaximumWidth(60)
        self.about_button.clicked.connect(self.show_about)
        selectors_layout.addStretch()
        selectors_layout.addWidget(self.about_button)

        # Add horizontal separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        # Add selectors_layout to main layout
        main_layout.addLayout(selectors_layout)
        main_layout.addWidget(separator)

        #######################################################################################
        # Matrix components
        #######################################################################################
        grid = QGridLayout()
        for i in range(6):
            for j in range(i, 6):
                field = QLineEdit()
                field.setPlaceholderText(f"C{i+1}{j+1}")
                self.coefficient_fields[(i, j)] = field
                field.textChanged.connect(self.update_dependent_fields)
                grid.addWidget(field, i, j)
        main_layout.addLayout(grid)

        #######################################################################################
        # Bottom panel
        #######################################################################################
        bottom_layout = QHBoxLayout()

        ############################################
        # Plotting options
        ############################################
        left_panel_layout = QVBoxLayout()

        # E, G or nu selector
        parameter_layout = QHBoxLayout()
        self.plotting_selector = QComboBox()
        self.plotting_selector.addItems(['Young modulus', 'Shear modulus', 'Poisson ratio', 'Linear compressibility'])
        self.plotting_selector.currentIndexChanged.connect(self.update_plotting_selectors)
        parameter_layout.addWidget(QLabel("Parameter:"))
        parameter_layout.addWidget(self.plotting_selector)
        left_panel_layout.addLayout(parameter_layout)

        # Plotting style
        style_layout = QHBoxLayout()
        self.plot_style_selector = QComboBox()
        self.plot_style_selector.addItems(['3D', 'Sections', 'Pole Figure'])
        self.plot_style_selector.currentIndexChanged.connect(self.update_plotting_selectors)
        style_layout.addWidget(QLabel("Plot type:"))
        style_layout.addWidget(self.plot_style_selector)
        left_panel_layout.addLayout(style_layout)

        # 'which' selector
        which_layout = QHBoxLayout()
        self.which_selector = QComboBox()
        self.which_selector.addItems(WHICH_OPTIONS.keys())
        which_layout.addWidget(QLabel("Value:"))
        which_layout.addWidget(self.which_selector)
        left_panel_layout.addLayout(which_layout)

        # Plot button
        self.calculate_button = QPushButton("Plot")
        self.calculate_button.clicked.connect(self.calculate_and_plot)
        left_panel_layout.addWidget(self.calculate_button)

        self.euler_button = QPushButton("Apply rotation")
        self.euler_button.setToolTip("Rotate stiffness tensor (Bunge ZXZ)")
        self.euler_button.clicked.connect(self.open_euler_dialog)
        left_panel_layout.addWidget(self.euler_button)

        # Add horizontal separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        left_panel_layout.addWidget(separator)

        ############################################
        # Numeric results
        ############################################
        self.result_labels = {}
        RESULT_GROUPS = {
            "Young modulus": [
                ("E_mean", "Mean"),
                ("E_voigt", "Voigt"),
                ("E_reuss", "Reuss"),
                ("E_hill", "Hill"),
            ],
            "Shear modulus": [
                ("G_mean", "Mean"),
                ("G_voigt", "Voigt"),
                ("G_reuss", "Reuss"),
                ("G_hill", "Hill"),
            ],
            "Poisson ratio": [
                ("nu_mean", "Mean"),
                ("nu_voigt", "Voigt"),
                ("nu_reuss", "Reuss"),
                ("nu_hill", "Hill"),
            ],
            "Linear compressibility (x1000)": [
                ("Beta_mean", "Mean"),
                ("Beta_voigt", "Voigt"),
                ("Beta_reuss", "Reuss"),
                ("Beta_hill", "Hill"),
            ],
            "Other": [
                ("K", "Bulk modulus"),
                ("Z", "Zener ratio"),
                ("A", "Univ. anisotropy factor"),
            ]
        }

        ############################################
        # Numeric results (grouped)
        ############################################
        self.result_labels = {}
        for group_name, items in RESULT_GROUPS.items():

            # Group title
            group_label = QLabel(group_name + ":")
            group_label.setStyleSheet("font-weight: bold; margin-top: 6px;")
            left_panel_layout.addWidget(group_label)

            # Indented layout
            indent_layout = QVBoxLayout()
            indent_layout.setContentsMargins(15, 0, 0, 0)

            for key, label_text in items:
                row = QHBoxLayout()

                label_name = QLabel(f"{label_text}:")
                label_value = QLabel("—")
                label_value.setMinimumWidth(100)
                label_value.setStyleSheet("font-family: Consolas, Courier;")

                self.result_labels[key] = label_value

                row.addWidget(label_name)
                row.addStretch()
                row.addWidget(label_value)

                indent_layout.addLayout(row)

            left_panel_layout.addLayout(indent_layout)

        # Fill space
        left_panel_layout.addStretch()


        bottom_layout.addLayout(left_panel_layout,1)

        ############################################
        # Plotting area
        ############################################
        # Add separator between the top and the bottom panels
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator2)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        bottom_layout.addWidget(self.canvas,4)

        #######################################################################################
        # Main widget
        #######################################################################################
        main_layout.addLayout(bottom_layout)
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        #######################################################################################
        # Initialize at load
        #######################################################################################
        self.symmetry_selector.setCurrentText('Triclinic')
        self.plotting_selector.setCurrentText('Young modulus')
        self.which_selector.setEnabled(False)

        self.C_matrix = np.zeros((6, 6))


    def update_fields(self):
        # Deactivate unused fields
        active_fields = self.selected_symmetry().required
        for (i, j), field in self.coefficient_fields.items():
            if (i, j) in active_fields:
                field.setEnabled(True)
                if field.text() == "0":
                    field.setText('')
            else:
                field.setEnabled(False)
                field.setText('0')

        # Turn on/off SG selection
        selected_symmetry_name = self.symmetry_selector.currentText()
        trig_or_tetra = selected_symmetry_name in ("Trigonal", "Tetragonal")
        self.point_group_selector.setEnabled(trig_or_tetra) # Turn on/off SG selector
        if trig_or_tetra:
            # Change list of possible SGs
            self.point_group_selector.setEnabled(True)
            space_groups = SYMMETRIES[selected_symmetry_name].keys()
            for i in range(len(space_groups)):
                self.point_group_selector.setItemText(i, list(space_groups)[i])
        self.diag_selector.setEnabled(selected_symmetry_name == "Monoclinic")

    def update_plotting_selectors(self):
        if (self.plotting_selector.currentText() == "Young modulus" or
            self.plotting_selector.currentText() == "Linear compressibility" or
                self.plot_style_selector.currentIndex() == 1):
            self.which_selector.setEnabled(False)
        else:
            self.which_selector.setEnabled(True)

    def calculate_and_plot(self):
        """Collect entries and compute the stiffness tensor"""
        coefficients = np.zeros((6, 6))
        for (i, j), field in self.coefficient_fields.items():
            try:
                coefficients[i, j] = float(field.text())
            except ValueError:
                coefficients[i, j] = 0
        C = np.array(coefficients)
        Csym = C + np.tril(C.T, -1) # Rebuild the lower triangular part

        try:
            stiff = StiffnessTensor(Csym)
            self.figure.clear()
            requested_value = self.plotting_selector.currentText()
            if requested_value == "Young modulus":
                value = stiff.Young_modulus
                plot_kwargs = {}
            elif requested_value == 'Linear compressibility':
                value = stiff.linear_compressibility
                plot_kwargs = {}
            else:
                if requested_value == 'Shear modulus':
                    value = stiff.shear_modulus
                else:
                    value = stiff.Poisson_ratio
                plot_kwargs = {'which': WHICH_OPTIONS[self.which_selector.currentText()]}
            if self.plot_style_selector.currentIndex() == 0:
                value.plot3D(fig=self.figure, **plot_kwargs)
            elif self.plot_style_selector.currentIndex() == 1:
                value.plot_xyz_sections(fig=self.figure)
            else:
                value.plot_as_pole_figure(fig=self.figure, **plot_kwargs)
            self.canvas.draw()
            if not np.all(self.C_matrix == Csym):
                self.result_labels["E_mean"].setText(f"{stiff.Young_modulus.mean():.3f}")
                self.result_labels["G_mean"].setText(f"{stiff.shear_modulus.mean():.3f}")
                self.result_labels["nu_mean"].setText(f"{stiff.Poisson_ratio.mean():.3f}")
                self.result_labels["Beta_mean"].setText(f"{stiff.linear_compressibility.mean()*1000:.3f}")
                for method in ['voigt', 'reuss', 'hill']:
                    C = stiff.average(method=method)
                    self.result_labels[f"E_{method}"].setText(f"{C.Young_modulus.eval([1,0,0]):.3f}")
                    self.result_labels[f"G_{method}"].setText(f"{C.shear_modulus.eval([1, 0, 0],[0,1,0]):.3f}")
                    self.result_labels[f"nu_{method}"].setText(f"{C.Poisson_ratio.eval([1, 0, 0], [0, 1, 0]):.3f}")
                    self.result_labels[f"Beta_{method}"].setText(f"{C.linear_compressibility.eval([1, 0, 0])*1000:.3f}")
                self.result_labels["K"].setText(f"{stiff.bulk_modulus:.3f}")
                try:
                    Z = stiff.Zener_ratio()
                    self.result_labels["Z"].setText(f"{stiff.Zener_ratio():.3f}")
                except ValueError:
                    self.result_labels["Z"].setText("—")
                self.result_labels["A"].setText(f"{stiff.universal_anisotropy:.3f}")
                self.C_matrix = Csym

        except ValueError as inst:
            QMessageBox.critical(self, "Singular stiffness", inst.__str__(), QMessageBox.Ok)


    def update_dependent_fields(self):
        symmetry = self.selected_symmetry()
        for equality in symmetry.equal:
            try:
                ref_value = float(self.coefficient_fields[equality[0]].text())
                for index in equality[1]:
                        self.coefficient_fields[index].setText(f"{ref_value}")
            except ValueError:
                pass
        for opposite in symmetry.opposite:
            try:
                ref_value = float(self.coefficient_fields[opposite[0]].text())
                for index in opposite[1]:
                        self.coefficient_fields[index].setText(f"{-ref_value}")
            except ValueError:
                pass
        if symmetry.C11_C12:
            try:
                C11 = float(self.coefficient_fields[(0, 0)].text())
                C12 = float(self.coefficient_fields[(0, 1)].text())
                for index in symmetry.C11_C12:
                    self.coefficient_fields[index].setText(f"{0.5*(C11-C12)}")
            except ValueError:
                pass

    def show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("About Elasticipy")
        dialog.setFixedWidth(400)

        layout = QVBoxLayout(dialog)

        # --- Logo ---
        here = Path(__file__).resolve().parent
        logo_path = here / "resources" / "logo_text.svg"

        if logo_path.exists():
            logo = QLabel()
            pixmap = QPixmap(str(logo_path))
            pixmap = pixmap.scaledToWidth(250, Qt.SmoothTransformation)
            logo.setPixmap(pixmap)
            logo.setAlignment(Qt.AlignCenter)
            layout.addWidget(logo)

        # --- Text ---
        text = QLabel(
            "A Python library for elasticity tensors computations<br><br>"
            "© 2024–2025 Dorian Depriester, MIT Licence"
        )
        text.setAlignment(Qt.AlignCenter)
        layout.addWidget(text)

        # --- Link ---
        link = QLabel(
            '<a href="https://elasticipy.readthedocs.io/">'
            'https://elasticipy.readthedocs.io/</a>'
        )
        link.setAlignment(Qt.AlignCenter)
        link.setOpenExternalLinks(True)
        layout.addWidget(link)

        # --- Bug report ---
        link = QLabel(
            '<a href="https://github.com/DorianDepriester/Elasticipy/issues">'
            'Report a bug</a>'
        )
        link.setAlignment(Qt.AlignCenter)
        link.setOpenExternalLinks(True)
        layout.addWidget(link)

        # --- Close button ---
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec_()

    def open_euler_dialog(self):
        if not hasattr(self, "euler_dialog"):
            self.current_config = {'sym': self.symmetry_selector.currentText(),
                              'pg': self.point_group_selector.currentText(),
                              'diag': self.diag_selector.currentText()}
            self.euler_dialog = EulerBungeDialog(self, C=self.C_matrix)
            self.euler_dialog.anglesChanged.connect(self.update_from_euler)
        self.euler_dialog.show()

    def update_from_euler(self, phi1, Phi, phi2):
        C0 = StiffnessTensor(self.C_matrix)  # StiffnessTensor initial
        rot = Rotation.from_euler('ZXZ', [phi1, Phi, phi2], degrees=True)
        if np.any(np.array([phi1, phi2, phi2])) != 0.:
            self.symmetry_selector.setCurrentText('Triclinic')
            C_new = C0.rotate(rot).matrix()
            for (i, j), field in self.coefficient_fields.items():
                self.coefficient_fields[(i,j)].setText(f"{C_new[i,j]}")
        else:
            for (i, j), field in self.coefficient_fields.items():
                self.coefficient_fields[(i,j)].setText(f"{self.euler_dialog.C[i,j]}")
            self.symmetry_selector.setCurrentText(self.current_config['sym'])
            self.point_group_selector.setCurrentText(self.current_config['pg'])
            self.diag_selector.setCurrentText(self.current_config['diag'])


def crystal_elastic_plotter():
    app = QApplication(sys.argv)
    try:
        here = Path(__file__).resolve().parent
        icon_path = here / "resources" / "favicon.png"
        icon = QIcon(str(icon_path))
        print(icon_path, icon_path.exists())
    except Exception:
        icon = QIcon()
    app.setWindowIcon(icon)
    window = ElasticityGUI()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec_())


from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QDoubleSpinBox, QPushButton
)
from qtpy.QtCore import Qt, Signal


class EulerBungeDialog(QDialog):
    anglesChanged = Signal(float, float, float)

    def __init__(self, parent=None, C=np.zeros((6, 6))):
        super().__init__(parent)
        self.setWindowTitle("Euler angles (Bunge ZXZ)")
        self._build_ui()
        self.reset()
        self.C = C

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
        for slider in self.sliders:
            slider.setValue(0)

    def _set_angles(self, phi1, Phi, phi2):
        for spin, val in zip(self.spins, (phi1, Phi, phi2)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._emit_angles()

    def _emit_angles(self):
        angles = [spin.value() for spin in self.spins]
        self.anglesChanged.emit(*angles)



if __name__ == "__main__":
    crystal_elastic_plotter()