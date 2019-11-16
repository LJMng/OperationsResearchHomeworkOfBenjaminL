'''
Created on 2019年11月7日

Homework on 2019.11.07

Simplex implementation based on web source implementation(https://www.jianshu.com/p/b233cfa06017). 
Changes have been made.

@author: Benjamin_L
'''

import copy
import numpy as np

def pivot():
    # 确定换入变量: 通过找最大值确定矩阵哪一列
    jnum = np.argmax(d[0, :-1])
    # 计算theta
    theta = []
    for i in range(1, bn):
        # 原程序这里d[i][jnum]的判断条件不正确，应为在满足>0的值中查找
        theta.append(d[i][-1] / d[i][jnum] if d[i][jnum] >0 else np.inf)
    print('theta: %s' % (['%.2f' % each for each in theta]) )
    # 确定换出变量
    inum = np.argmin(theta)
    # 基于换入与换出更新基变量
    s[inum] = jnum
    r = d[inum+1][jnum]
    # 旋转运算
    d[inum+1] /= r
    for i in [x for x in range(1, bn) if d[x][jnum]!=0 and x!=inum+1]:
        d[i, :] -= d[inum+1, :] * d[i, jnum]

def update_sigma():
    # 计算新表的sigma
    for c in range(cn):
        d[0][c] = factors[c] - sum(factors[s] * d[1:, c])
        
def solve():
    count = 0
    while True:
        count = count +1 
        print('LOOP', count)
        update_sigma()
        print('基变量: %s' % ['x%d'%v for v in s])
        print('Simplex matrix:')
        print(d)
        print('sigma: %s' % (['%.2f' % each for each in d[0][:-1]]) )
        pivot()
        if (d[0, :-1]<=0).all():   break
        print()
            
def printSol():
    print()
    for i in range(cn - 1):
        if i in s:
            print("x"+str(i)+"=%.2f" % d[s.index(i)+1][-1])
        else:
            print("x"+str(i)+"=0.00")
    print("objective is %.2f"%(-d[0][-1]))
    
if __name__=='__main__':
    d = np.loadtxt("data2.txt", dtype=np.float)
    (bn, cn) = d.shape
    # 基变量列表
    s = list(range(cn-bn, cn-1))    
    # 保存各变量系数，用于后面计算
    factors = copy.deepcopy(d[0])
    
    solve()
    printSol()