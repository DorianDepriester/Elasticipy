import sys

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QGridLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Elasticipy.CrystalSymmetries import SYMMETRIES
from Elasticipy.FourthOrderTensor import StiffnessTensor

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
        self.plotting_selector.addItems(['Young modulus', 'Shear modulus', 'Poisson ratio'])
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

        # Fill space
        left_panel_layout.addStretch()
        bottom_layout.addLayout(left_panel_layout)

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
        bottom_layout.addWidget(self.canvas)

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

        if np.abs(np.linalg.det(Csym)) < 1e-5:
            self.raise_singularity_error()
        else:
            stiff = StiffnessTensor(Csym)

            self.figure.clear()
            requested_value = self.plotting_selector.currentText()
            if requested_value == "Young modulus":
                value = stiff.Young_modulus
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

    def raise_singularity_error(self):
        """Print a message error if the stiffness tensor is singular"""
        error_message = "The stiffness tensor you entered is singular."
        QMessageBox.critical(self, "Singular stiffness", error_message, QMessageBox.Ok)

def crystal_elastic_plotter():
    app = QApplication(sys.argv)
    window = ElasticityGUI()
    window.show()
    sys.exit(app.exec_())
