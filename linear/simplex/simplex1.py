"""
Scipy.optimize example.
"""

import numpy as np

z = np.array([-8, -1, -1])
a = np.array([[1, 7, 7], [6, 5, 5]])
b = np.array([10,9])
x1_bound = x2_bound = x3_bound =(0, None)

from scipy import optimize
res = optimize.linprog(z, A_ub=a, b_ub=b,bounds=(x1_bound, x2_bound, x3_bound))
print(res)