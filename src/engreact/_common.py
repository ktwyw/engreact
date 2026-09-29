"""Small helpers shared by the modules."""

import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz  # np.trapezoid arrived in NumPy 2.0; np.trapz left in 2.4
