import numpy as np



v1 = np.arange(3*2).reshape((2,3)).transpose()
v2 = np.arange(2).reshape(2)

v2[0] = 1
v2[0] = 2

print(v1)
print(v2)
print(np.dot(v1, v2))