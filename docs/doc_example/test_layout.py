import numpy as np 
# NCHW = 3 32 5 5   ->padding补齐
a = np.zeros((3,32,5,5))

for i in range(32) :
    a[0][i][0][0] = i
    a[1][i][1][1] = i * 10

b = a.reshape((3,2,16,5,5))

c = b.transpose()