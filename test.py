import numpy as np


# v1 = np.arange(3*2).reshape((2,3)).transpose()
# v2 = np.arange(2).reshape(2)

# v1 = np.random.randn(3,2,2)
# v2 = np.pad(v1,((0,0),(1,1),(1,1)),mode='constant',constant_values=(0))


# v1 = np.random.randn(4,3,3)

# v2 = np.reshape(v1, (4,-1))

# v1 = np.arange(5).reshape(5,1)
# v2 = np.arange(5).reshape(1,5)

# v1 = np.random.randn(10,500).T
# v2 = np.random.randn(10)

v1 = np.arange(36).reshape(2,2,3,3)


v1[0,0,1,1] = 12

v2 = np.arange(9).reshape(3,3)
v = (1,1)
print(v2[v])
# print(v1[0,0])
# a = np.where(v1[0,0] == v1[0,0].max())

# print(a[0] + a[1])
# print(v1[:,:,::-1,::-1])
# print(np.flip(v1))

# print(v2)
# print(np.dot(v1, v2).shape)
