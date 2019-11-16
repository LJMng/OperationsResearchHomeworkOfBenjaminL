import numpy as np

r = ('A1', 'A2', 'A3')  # 产地
c = ('B1', 'B2', 'B3', 'B4')  # 销地
# (产地,销地):运价
x = [[6, 2, 6, 0],
     [4, 9, 5, 0],
     [5, 2, 1, 0],
     ]
datas = dict()
for i in r:
	for j in c:
		datas[i, j] = x[r.index(i)][c.index(j)]

# 产地:产量
y = [60, 55, 51]
datac = dict()
for i in r:
	datac[i] = y[r.index(i)]

# 销地:销量
z = [35, 37, 22, 72]
datax = dict()
for j in c:
	datax[j] = z[c.index(j)]

np.save('a_datas.npy', datas)
np.save('a_datac.npy', datac)
np.save('a_datax.npy', datax)

a = np.load('a_datas.npy')
b = np.load('a_datac.npy')
c = np.load('a_datax.npy')

print("(产地,销地):运价\n", a)
print("\n产地:产量\n", b)
print("\n销地:销量\n", c)

from pymprog import *
import numpy as np

begin('Transport')

datas = np.load('a_datas.npy').tolist()  # (产地,销地):运价
datac = np.load('a_datac.npy').tolist()  # 产地:产量
datax = np.load('a_datax.npy').tolist()  # 销地:销量
x = var('x', datas.keys())                # 调运方案

minimize(sum(datas[i, j]*x[i, j] for (i, j) in datas.keys()), 'Cost')  # 总运费最少
for i in datac.keys():    # 产地产量约束
    sum(x[i, j] for j in datax.keys()) == datac[i]
for j in datax.keys():   # 销地销量约束
    sum(x[i ,j] for i in datac.keys()) == datax[j]

def report():
    print("调运方案(最优之一)")
    for (i, j) in datas.keys():
        if x[i, j].primal > 0 and datas[i, j] != 0:
            print("产地:%s -> 销地:%s 运输量:%-2d 运价:%2d" % (i, j, int(x[i, j].primal), int(datas[i, j])))
    print("总费用:%d"%int(vobj()))

solve()
report()

end()