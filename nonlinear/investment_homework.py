'''
Created on 2019年11月14日

A python script for investment homework in non-linear problems.

@author: Benjamin_L
'''

import numpy as np


def exhaustion_search_1(max_func, constraints, x_candidates):
    index = np.argmax([max_func(x) for x in x_candidates if constraints(x)])
    return x_candidates[index]

def exhaustion_search_2(max_func, constraints, x_range=[[0, 3], [0, 4.5]], x_step=0.1):
    max_func_value, max_x_values = None, None
    for x1 in list(np.arange(x_range[0][0], x_range[0][1]+x_step, x_step)):
        for x2 in list(np.arange(x_range[1][0], x_range[1][1]+x_step, x_step)):
            if not constraints([x1, x2]):
                break
            else:
                x_values = [x1, x2]
                calculated = max_func(x_values)
                if max_func_value is None:
                    max_x_values, max_func_value = x_values, calculated
                elif max_func_value<calculated:
                    max_x_values, max_func_value = x_values, calculated
    return max_x_values

def using_scipy(min_func, scipy_cons, init_x):
    from scipy.optimize import minimize
    return minimize(min_func, init_x, constraints=scipy_cons)
    

if __name__ == '__main__':
    # z = x1 + x2^2
    max_func = lambda x: x[0]+(x[1]**2)
    # s.t. 1 3*x1+2*x2<=9
    # s.t. 2 x[i] >=0 (i=1,2)
    constraints = lambda x: 3*x[0]+2*x[1]<=9 and (np.array(x)>=0).all()
    # x: x1=[0, 3], x2=[0, 4.5]
    step = 0.1
    x = [ [i, j] for i in list(np.arange(0, 3+step, step)) for j in list(np.arange(0, 4.5+step, step))]
    result1 = exhaustion_search_1(max_func, constraints, x)
    print(result1)
    result2 = exhaustion_search_2(max_func, constraints)
    print(result2)

    # z = x1 + x2^2
    min_func = lambda x: -(x[0]+(x[1]**2))
    # s.t.
    scipy_cons = [
        # s.t. 1 3*x1+2*x2<=9 <=> 9-(3*x1+2*x2) >= 0
        {'type': 'ineq', 'fun': lambda x: -(3*x[0]+2*x[1]) + 9},
        # s.t. 2 x[i] >=0 (i=1,2)
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1]},
    ]
    scipy_init_x = [0.5 for _ in range(2)]
    result3 = using_scipy(min_func, scipy_cons, scipy_init_x)
    print(result3)