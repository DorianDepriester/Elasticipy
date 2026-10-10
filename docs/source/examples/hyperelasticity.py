"""
==============================================
Fit an hyperelastic model on a tensile curve
==============================================

This example shows how to fit an hyperelastic model on a tensile curve.
"""
###############################################################################
# Create the hyper-elastic model
# -------------------------------------


###############################################################################
# Simulate the tensile curve
# -------------------------------------
# We consider the Yeoh model
from elasticipy.hyperelasticity import Yeoh
C = [39.8, -131.9, 82.8] # MPa
yh = Yeoh(C) # Incompressible case

###############################################################################
# Compute the corresponding tensile curve
import numpy as np
n = 100
stretch = np.linspace(0,1, n) # 0 to 100% elongation
tensile_stress = yh.tensile_curve(stretch)

###############################################################################
# and add some noise
noise = np.random.randn(n) * 50
tensile_stress = tensile_stress + noise

###############################################################################
# Fit a Yeoh model on the simulated sample
# ----------------------------------------
yh_fit = Yeoh.fit(stretch, tensile_stress)

###############################################################################
# Check fitted parameters
print(yh_fit.C) # To be compared with C (see above)

###############################################################################
# Now recalculate the stress from the fitted model
tensile_stress_fit = yh_fit.tensile_curve(stretch)

###############################################################################
# and plot sample data and fitted model
# -------------------------------------
import matplotlib.pyplot as plt
plt.plot(stretch, tensile_stress,       label='Sample (Yeoh + noise)')
plt.plot(stretch, tensile_stress_fit,   label='Fitted Yeoh')
plt.legend()
plt.xlabel('True strain')
plt.ylabel('Stress (MPa)')
plt.title('Fitting a Yeoh hyperelastic model')
plt.show()
