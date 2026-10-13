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
C = [0.18, -2e-3, 5e-5] # MPa, taken from https://help.altair.com/hwsolvers/rad/topics/solvers/rad/yeoh_hyperelastic_material_verification_intro_r.htm
yh = Yeoh(C) # Incompressible case

###############################################################################
# Compute the corresponding tensile curve
import numpy as np
n = 500
stretch = np.linspace(-0.8,7, n) # 0 to 100% elongation
tensile_stress = yh.tensile_curve(stretch)

###############################################################################
# and add some noise
noise = np.random.randn(n) * 1.0
tensile_stress_noise = tensile_stress + noise

###############################################################################
# Fit a Yeoh model on the simulated sample
# ----------------------------------------
yh_fit = Yeoh.fit(stretch, tensile_stress_noise)

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
plt.plot(stretch, tensile_stress, label='Ground truth (Yeoh)')
plt.scatter(stretch, tensile_stress_noise, label='Sample (Yeoh + noise)', marker='.')
plt.plot(stretch, tensile_stress_fit,   label='Fitted Yeoh', linestyle='dotted')
plt.legend()
plt.xlabel('Engineering strain')
plt.ylabel('Engineering stress (MPa)')
plt.title('Fitting a Yeoh hyperelastic model')
plt.show()
