'''
Created on 2019年11月27日

@author: Benjamin_L
'''

import numpy as np

from scipy.optimize import minimize

class Homework_20191121():
    @staticmethod
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
    
    @staticmethod
    def execute():
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
        scipy_init_x = Homework_20191121.init_x()
        print('init x:', scipy_init_x)
        result = minimize(min_func, scipy_init_x, constraints=scipy_cons)
        print(result)
    
class Homework_20191128():
    @staticmethod
    def execute():
        # suppose a=4.41176, b=2.04081633, c=0.35971223:
#         a = 4.41176
#         b = 2.04081633 
#         c = 0.35971223
        a = 4.41
        b = 2.04
        c = 0.36
        #     max f(x)=x1(625-ax1)+ x2(300-bx2)+ x3(100-cx3)
        #        ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
        #     min f(x)=-(x1(625-ax1)+ x2(300-bx2)+ x3(100-cx3))
        min_func = lambda x: -x[0]*(625-a*x[0]) - x[1]*(300-b*x[1]) - x[2]*(100-c*x[2])
        # s.t.
        # 1) x1^2-x2^2+x3^3<=10
        # 2) x1^3+x2^2+4x3^3>=20
        # s.t.
        scipy_cons = [
            # 1) 70-x1<=0 => x1-70  >=0
            # 2) x1-90<=0 => 90-x1 >=0
            {'type': 'ineq', 'fun': lambda x:  x[0]-70},
            {'type': 'ineq', 'fun': lambda x:  90-x[0]},
            # 3) 90-x2<=0 => x2-90  >=0
            # 4) x2-110<=0 => 110-x2  >=0
            {'type': 'ineq', 'fun': lambda x:  x[1]-90},
            {'type': 'ineq', 'fun': lambda x:  110-x[1]},
            # 5) 120-x3<=0 => x3-120  >=0
            # 6) x3-149<=0 => 149-x3  >=0
            {'type': 'ineq', 'fun': lambda x:  x[2]-120},
            {'type': 'ineq', 'fun': lambda x:  149-x[2]},
            # 7) 575-ax1-bx2-cx3<=0 => ax1+bx2+cx3-575  >=0
            {'type': 'ineq', 'fun': lambda x:  a*x[0]+b*x[1]+c*x[2]-575},
        ]
        scipy_init_x = [0, 0, 0]
        print('init x:', scipy_init_x)
        result = minimize(min_func, scipy_init_x, constraints=scipy_cons)
        print(result)
    

if __name__ == '__main__':
    Homework_20191128.execute()