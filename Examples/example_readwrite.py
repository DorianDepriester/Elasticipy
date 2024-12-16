from Elasticipy.FourthOrderTensor import StiffnessTensor

C = StiffnessTensor.from_MP("mp-1048")
C.save_to_txt("TiNi.txt")
C.save_to_txt("TiNi.txt", matrix_only=True)

C2 = StiffnessTensor.from_txt_file("TiNi.txt")
Cu2= StiffnessTensor.from_txt_file("TiNi.txt")


