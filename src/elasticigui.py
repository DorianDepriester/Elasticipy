import sys
from time import monotonic

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QComboBox, QGridLayout, QLabel,
    QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from Elasticipy.FourthOrderTensor import StiffnessTensor

class SymmetryRelationships:
    def __init__(self, active_cells=(), equal_cells=(), opposite_cells=(), halfC11_C12=()):
        self.active = active_cells
        self.equal = equal_cells
        self.opposite = opposite_cells
        self.half = halfC11_C12

isotropic=SymmetryRelationships(active_cells=[(0, 0), (0, 1)],
                                equal_cells=[((0, 0), [(1, 1), (2, 2)]),
                                             ((0, 1), [(0, 2), (1, 2)]),
                                             ((3, 3), [(4, 4), (5, 5)])],
                                halfC11_C12=[(3, 3), (4, 4), (5, 5)])
cubic=SymmetryRelationships(active_cells=[(0, 0), (0, 1), (3, 3)],
                            equal_cells=[((0, 0), [(1, 1), (2, 2)]),
                                         ((0, 1), [(0, 2), (1, 2)]),
                                         ((3, 3), [(4, 4), (5, 5)])])
hexagonal=SymmetryRelationships(active_cells=[(0, 0), (0, 1), (0, 2), (2, 2), (3, 3)],
                                equal_cells=[((0, 0), [(1, 1)]),
                                             ((0, 2), [(1, 2)]),
                                             ((3, 3), [(4, 4)])],
                                halfC11_C12=[(5, 5)])
tetragonal_1=SymmetryRelationships(active_cells=[(0, 0), (0, 1), (0, 2), (0, 5), (2, 2), (3, 3), (5, 5)],
                                   equal_cells=[((0, 0), [(1, 1)]),
                                                ((0, 2), [(1, 2)]),
                                                ((3, 3), [(4, 4)])],
                                   opposite_cells=[((0, 5), [(1, 5)])])
tetragonal_2=SymmetryRelationships(active_cells=[(0, 0), (0, 1), (0, 2), (2, 2), (3, 3), (5, 5)],
                                   equal_cells=[((0, 0), [(1, 1)]),
                                                ((0, 2), [(1, 2)]),
                                                ((3, 3), [(4, 4)])])
trigonal_1=SymmetryRelationships(active_cells=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (2, 2), (3, 3)],
                                 equal_cells=[((0, 0), [(1, 1)]),
                                              ((0, 2), [(1, 2)]),
                                              ((0, 3), [(4, 5)]),
                                              ((3, 3), [(4, 4)]),],
                                 opposite_cells=[((0, 3), [(1, 3)]),
                                                 ((0, 4), [(1, 4), (3, 5)]),],
                                 halfC11_C12=[(5, 5)])
trigonal_2=SymmetryRelationships(active_cells=[(0, 0), (0, 1), (0, 2), (0, 3), (2, 2), (3, 3)],
                                 equal_cells=[((0, 0), [(1, 1)]),
                                              ((0, 2), [(1, 2)]),
                                              ((3, 3), [(4, 4)]),
                                              ((0, 3), [(4, 5)])],
                                 opposite_cells=[((0, 3), [(1, 3)])],
                                 halfC11_C12=[(5, 5)])
orthorhombic=SymmetryRelationships(active_cells=[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4), (5, 5)])
active_cell_monoclinic_0 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4), (5, 5)]
monoclinic1=SymmetryRelationships(active_cells= active_cell_monoclinic_0 + [(0, 4), (1, 4), (2, 4), (3, 5)])
monoclinic2=SymmetryRelationships(active_cells= active_cell_monoclinic_0 + [(0, 5), (1, 5), (2, 5), (3, 4)])
triclinic=SymmetryRelationships(active_cells=[(i, j) for i in range(6) for j in range(6)])

SYMMETRIES = {'Isotropic': isotropic,
              'Cubic': cubic,
              'Hexagonal': hexagonal,
              'Tetragonal': [tetragonal_1, tetragonal_2],
              'Trigonal': [trigonal_1, trigonal_2],
              'Orthorhombic': orthorhombic,
              'Monoclinic': [monoclinic1,  monoclinic2],
              'Triclinic': triclinic}

SPACE_GROUPS = {'Trigonal':   ["3, -3", "32, -3m, 3m"],
                'Tetragonal': ["4, -4, 4/m", "4mm, -42m, 422, 4/mmm"]}

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
            space_group_index = self.space_group_selector.currentIndex()
            symmetry = symmetry[space_group_index]
        elif symmetry_name == "Monoclinic":
            diad_index = self.diag_selector.currentIndex()
            symmetry = symmetry[diad_index]
        return symmetry

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()
        # Symmetry, Space Group, and Diad Selection (aligned horizontally)
        selectors_layout = QHBoxLayout()

        # Symmetry selection
        self.symmetry_selector = QComboBox()
        self.symmetry_selector.addItems(SYMMETRIES.keys())
        self.symmetry_selector.currentIndexChanged.connect(self.update_fields)
        selectors_layout.addWidget(QLabel("Crystal symmetry:"))
        selectors_layout.addWidget(self.symmetry_selector)

        # Space Group selection
        self.space_group_selector = QComboBox()
        self.space_group_selector.addItems(['', ''])
        self.space_group_selector.currentIndexChanged.connect(self.update_fields)
        selectors_layout.addWidget(QLabel("Space group symmetry:"))
        selectors_layout.addWidget(self.space_group_selector)

        # Diad selection
        self.diag_selector = QComboBox()
        self.diag_selector.addItems(["diad || x2", "diad || x3"])
        self.diag_selector.currentIndexChanged.connect(self.update_fields)
        selectors_layout.addWidget(QLabel("Diad convention:"))
        selectors_layout.addWidget(self.diag_selector)

        # Ajouter un séparateur horizontal
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)  # Ligne horizontale
        separator.setFrameShadow(QFrame.Sunken)  # Optionnel : style ombré

        # Add selectors_layout to main layout
        main_layout.addLayout(selectors_layout)
        main_layout.addWidget(separator)

        # Matrix component
        grid = QGridLayout()
        for i in range(6):
            for j in range(i, 6):
                field = QLineEdit()
                field.setPlaceholderText(f"C{i+1}{j+1}")
                self.coefficient_fields[(i, j)] = field
                field.textChanged.connect(self.update_dependent_fields)
                grid.addWidget(field, i, j)
        main_layout.addLayout(grid)

        # Plot button
        self.calculate_button = QPushButton("Plot")
        self.calculate_button.clicked.connect(self.calculate_and_plot)
        main_layout.addWidget(self.calculate_button)

        # Display area
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

        # Main widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def update_fields(self):
        # Deactivate unused fields
        active_fields = self.selected_symmetry().active
        for (i, j), field in self.coefficient_fields.items():
            if (i, j) in active_fields:
                field.setEnabled(True)
            else:
                field.setEnabled(False)
                field.setText('')

        # Turn on/off SG selection
        selected_symmetry_name = self.symmetry_selector.currentText()
        if selected_symmetry_name in ("Trigonal", "Tetragonal"):
            # Turn on and change list of possible SGs
            self.space_group_selector.setEnabled(True)
            for i in range(2):
                self.space_group_selector.setItemText(i, SPACE_GROUPS[selected_symmetry_name][i])
        else:
            # Turn off
            self.space_group_selector.setEnabled(False)
        self.diag_selector.setEnabled(selected_symmetry_name == "Monoclinic")


    def calculate_and_plot(self):
        """Collect entries and compute the stiffness tensor"""
        coefficients = np.zeros((6, 6))
        for (i, j), field in self.coefficient_fields.items():
            try:
                coefficients[i, j] = float(field.text())
            except ValueError:
                coefficients[i, j] = 0

        C = np.array(coefficients)
        stiff = StiffnessTensor(C + np.tril(C.T, -1))

        E = stiff.Young_modulus
        self.figure.clear()
        E.plot3D(fig=self.figure)
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
        if symmetry.half:
            try:
                C11 = float(self.coefficient_fields[(0, 0)].text())
                C12 = float(self.coefficient_fields[(0, 1)].text())
                for index in symmetry.half:
                    self.coefficient_fields[index].setText(f"{0.5*(C11-C12)}")
            except ValueError:
                pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ElasticityGUI()
    window.show()
    sys.exit(app.exec_())
