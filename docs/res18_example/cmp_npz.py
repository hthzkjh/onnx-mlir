import numpy as np
import sys

if len(sys.argv) != 3:
    print("error! must give two npz to compare")
    quit()
a = sys.argv[1]
b = sys.argv[2]
c = np.load(a)['output']
d = np.load(b)['output']

if c.shape != d.shape :
    print("shape not compare")
    quit()

print(c.shape)

for i in range(len(c)):
    for j in range(len(c[i])):
        for k in range(len(c[i][j])):
            if c[i][j][k] - d[i][j][k] > 1e-5:
                print(str(i)+"-"+str(j)+"-"+str(k)+":"+str(c[i][j][k])+"  "+str(d[i][j][k]))
                print("not the same")

print("same npz")

