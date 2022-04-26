import numpy as np

a0 = np.array([1, 2, 3])
# numpy array->1
#              2
#              3

a1 = np.zeros(2)
a2 = np.ones(3)
a3 = np.empty(4)
a4 = np.arange(2, 9, 3)
a5 = np.linspace(1, 5, 4)

# adding, removing, sorting elements
b0 = np.array([1, 3, 2, 5, 4])
# b0 = np.sort(b0)
b1 = np.array([2, 5, 7])
# b2 = np.array([2, 5, 6, 7, 8, 9, 0, 3])
b3 = np.concatenate((b0, b1))
# b4 = np.concatenate((b3, b2), axis=0)

b3 = b3.reshape(-1, 8)

# convert a 1D array into a 2D array
c0 = np.array([1, 2, 3])
c1 = c0[np.newaxis, :]
c2 = np.expand_dims(c0, axis=0)

# indexing and slicing

d0 = np.nonzero(c0 <= 2)
print(c0[d0])  # ->array([1, 2])

# create array from existing data

