'''
Created on 2019年11月7日

Homework on 2019.11.07

Simplex implementation based on web source implementation(https://www.jianshu.com/p/b233cfa06017). 
Changes have been made.

@author: Benjamin_L
'''

import numpy as np

def pivot():
    l = list(d[0][:-2])
    jnum = l.index(max(l)) #转入编号
    m = []
    for i in range(bn):
        if d[i][jnum] == 0:
            m.append(0.)
        else:
            m.append(d[i][-1]/d[i][jnum])
    # 原程序这里d[i][jnum]的判断条件不正确，应为在满足>0的值中查找
    inum = m.index(min([x for x in m[1:] if x>0]))  #转出下标
    s[inum-1] = jnum
    r = d[inum][jnum]
    d[inum] /= r
    for i in [x for x in range(bn) if x !=inum]:
        r = d[i][jnum]
        d[i] -= r * d[inum]
        
def solve():
    flag = True
    while flag:
        print(d)
        if max(list(d[0][:-1])) <= 0: #直至所有系数小于等于0
            flag = False
        else:
            pivot()
            
def printSol():
    for i in range(cn - 1):
        if i in s:
            print("x"+str(i)+"=%.2f" % d[s.index(i)+1][-1])
        else:
            print("x"+str(i)+"=0.00")
    print("objective is %.2f"%(-d[0][-1]))
    
    
if __name__=='__main__':
    d = np.loadtxt("data2.txt", dtype=np.float)
    (bn,cn) = d.shape
    s = list(range(cn-bn,cn-1)) #基变量列表
    solve()
    printSol()