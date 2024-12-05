class SymmetryRelationships:
    def __init__(self, required_cells=(), equal_cells=(), opposite_cells=(), C11_C12=()):
        self.required = required_cells
        self.equal = equal_cells
        self.opposite = opposite_cells
        self.C11_C12 = C11_C12


isotropic=SymmetryRelationships(required_cells=[(0, 0), (0, 1)],
                                equal_cells=[((0, 0), [(1, 1), (2, 2)]),
                                             ((0, 1), [(0, 2), (1, 2)]),
                                             ((3, 3), [(4, 4), (5, 5)])],
                                C11_C12=[(3, 3), (4, 4), (5, 5)])
cubic=SymmetryRelationships(required_cells=[(0, 0), (0, 1), (3, 3)],
                            equal_cells=[((0, 0), [(1, 1), (2, 2)]),
                                         ((0, 1), [(0, 2), (1, 2)]),
                                         ((3, 3), [(4, 4), (5, 5)])])
hexagonal=SymmetryRelationships(required_cells=[(0, 0), (0, 1), (0, 2), (2, 2), (3, 3)],
                                equal_cells=[((0, 0), [(1, 1)]),
                                             ((0, 2), [(1, 2)]),
                                             ((3, 3), [(4, 4)])],
                                C11_C12=[(5, 5)])
tetragonal_1=SymmetryRelationships(required_cells=[(0, 0), (0, 1), (0, 2), (0, 5), (2, 2), (3, 3), (5, 5)],
                                   equal_cells=[((0, 0), [(1, 1)]),
                                                ((0, 2), [(1, 2)]),
                                                ((3, 3), [(4, 4)])],
                                   opposite_cells=[((0, 5), [(1, 5)])])
tetragonal_2=SymmetryRelationships(required_cells=[(0, 0), (0, 1), (0, 2), (2, 2), (3, 3), (5, 5)],
                                   equal_cells=[((0, 0), [(1, 1)]),
                                                ((0, 2), [(1, 2)]),
                                                ((3, 3), [(4, 4)])])
trigonal_1=SymmetryRelationships(required_cells=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (2, 2), (3, 3)],
                                 equal_cells=[((0, 0), [(1, 1)]),
                                              ((0, 2), [(1, 2)]),
                                              ((0, 3), [(4, 5)]),
                                              ((3, 3), [(4, 4)]),],
                                 opposite_cells=[((0, 3), [(1, 3)]),
                                                 ((0, 4), [(1, 4), (3, 5)]),],
                                 C11_C12=[(5, 5)])
trigonal_2=SymmetryRelationships(required_cells=[(0, 0), (0, 1), (0, 2), (0, 3), (2, 2), (3, 3)],
                                 equal_cells=[((0, 0), [(1, 1)]),
                                              ((0, 2), [(1, 2)]),
                                              ((3, 3), [(4, 4)]),
                                              ((0, 3), [(4, 5)])],
                                 opposite_cells=[((0, 3), [(1, 3)])],
                                 C11_C12=[(5, 5)])
orthorhombic=SymmetryRelationships(
    required_cells=[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4), (5, 5)])
active_cell_monoclinic_0 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2), (3, 3), (4, 4), (5, 5)]
monoclinic1=SymmetryRelationships(required_cells=active_cell_monoclinic_0 + [(0, 4), (1, 4), (2, 4), (3, 5)])
monoclinic2=SymmetryRelationships(required_cells=active_cell_monoclinic_0 + [(0, 5), (1, 5), (2, 5), (3, 4)])
triclinic=SymmetryRelationships(required_cells=[(i, j) for i in range(6) for j in range(i, 6)])

SYMMETRIES = {'Isotropic': isotropic,
              'Cubic': cubic,
              'Hexagonal': hexagonal,
              'Tetragonal': {"4, -4, 4/m": tetragonal_1, "4mm, -42m, 422, 4/mmm":tetragonal_2},
              'Trigonal': {"3, -3": trigonal_1, "32, -3m, 3m":trigonal_2},
              'Orthorhombic': orthorhombic,
              'Monoclinic': {"Diad || y": monoclinic1, "Diad || z": monoclinic2},
              'Triclinic': triclinic}
