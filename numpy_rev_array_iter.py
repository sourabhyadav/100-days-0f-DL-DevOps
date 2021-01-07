import numpy as np

# Iterating over a numpy array using nditer function

# The order of iteration is chosen to match the memory layout of an array, without considering a particular ordering.
a = np.arange(0, 60, 5)
a = a.reshape(3,4)
print("Orignial array: ", a)
print("Row-wise iteration")
for x in np.nditer(a): 
    print(x)            

b = a.T             # Transpose of a
print("Transpose of a: ", b)
print("Column-wise iteration")
for y in np.nditer(b):
    print(y)


# Defining the order of iteration
print("Original array: ", a)
print("Row-wise C-Style")
for x in np.nditer(a, order = 'C'):         # Row-wise
    print(x)

print("Row-wise F-Style")
for x in np.nditer(a, order = 'F'):         # Column-wise
    print(x)


# TODO: nditer can be used to modify the values for ndarry in a very short manner