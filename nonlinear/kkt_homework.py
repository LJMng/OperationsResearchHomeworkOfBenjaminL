'''
Created on 2019年11月27日

@author: Benjamin_L
'''

import numpy as np

from scipy.optimize import minimize

def init_x():
    x1 = list(np.arange(0, 50, 0.5))
    x2 = list(np.arange(-5, 50, 0.5))
    x3 = list(np.arange(0, 50, 0.5))
    
    constaints_func = lambda x: x[0]**2 - x[1]**2 + (x[2]**3) <=10 and\
                                x[0]**3 + x[1]**2 + 4*(x[2]**3) >=20
    
    for v1 in x1:
        for v2 in x2:
            for v3 in x3:
                fit = constaints_func((v1, v2, v3))
                if fit:
                    return v1, v2, v3
    raise Exception("Fail to initiate x.")
    
    

if __name__ == '__main__':
    # min f(x)=x1^4+x2^2+5x1x2x3
    # if exception 'Positive directional derivative for linesearch' is thrown, please divide a number to make z smaller.
    # if exception 'Inequality constraints incompatible' is thrown, please adjust the initiated x.
    min_func = lambda x: x[0]**4+x[1]**2+5*x[0]*x[1]*x[2]
    # s.t.
    # 1) x1^2-x2^2+x3^3<=10
    # 2) x1^3+x2^2+4x3^3>=20
    # s.t.
    scipy_cons = [
        # 1) x1^2-x2^2+4x3^3<=10 => 10-x1^2+x2^2-4x3^3  >=0
        {'type': 'ineq', 'fun': lambda x:  10 - x[0]**2 + x[1]**2 - (x[2]**3)},
        # 2) x1^3+x2^2+4x3^3>=20 => -20+x1^3+x2^2+4x3^3 >=0
        {'type': 'ineq', 'fun': lambda x: -20 + x[0]**3 + x[1]**2 + 4*(x[2]**3)},
    ]
    scipy_init_x = init_x()
    print('init x:', scipy_init_x)
    result = minimize(min_func, scipy_init_x, constraints=scipy_cons)
    print(result)