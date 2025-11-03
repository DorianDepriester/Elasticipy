from Elasticipy.tensors.elasticity import StiffnessTensor

C = StiffnessTensor.from_MP("mp-1048")
C.save_to_txt("TiNi.txt")
C.save_to_txt("TiNi-matrix.txt", matrix_only=True)

C2 = StiffnessTensor.from_txt_file("TiNi.txt")
C3 = StiffnessTensor.from_txt_file("TiNi-matrix.txt")
print(C==C2)
print(C==C3)


