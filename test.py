import numpy as np


# v1 = np.arange(3*2).reshape((2,3)).transpose()
# v2 = np.arange(2).reshape(2)

# v1 = np.random.randn(3,2,2)
# v2 = np.pad(v1,((0,0),(1,1),(1,1)),mode='constant',constant_values=(0))


v1 = np.random.randn(4,3,3)

v2 = np.reshape(v1, (4,-1))



print(v1)
print(v2)
# print(np.dot(v1, v2))
