import numpy as np

'''
# no copy: numpy.array = numpy.array
# Simple assignments do not make the copy of array object. Instead, it uses the same id() of the original array to access it. The id() returns a universal identifier of Python object, similar to the pointer in C.

a = np.arange(9)
print("id of a: ", id(a))
b = a
print("id of b: ", id(b))
b.shape = 3,3
b[0][0] = 10
print("reshaped b: \n", b)
print("new a: \n", a)
'''

'''
# shallow copy: numpy.view()
# NumPy has ndarray.view() method which is a new array object that looks at the same data of the original array. Unlike the earlier case, change in dimensions of the new array doesn’t change dimensions of the original.
a = np.arange(9)
print("id of a: ", id(a))
b = a.view()
print("id of b: ", id(b))
b.shape = 3,3
b[0][0] = 10
print("reshaped b: \n", b)
print("new a: \n", a)
# Note: Here the shape and id of both the arrays are different but assignment refelected in both the arrays
''' 

'''
# shallow copy 2: array slicing
a = np.array([[10,10], [2,3], [4,5]]) 
print("a: \n", a)
s = a[:, :2]
print("id of a: ", id(a))
print("id of s: ", id(s))
print("s: \n", s)
'''

# deep copy
# The ndarray.copy() function creates a deep copy. It is a complete copy of the array and its data, and doesn’t share with the original array.
a = np.arange(9)
print("id of a: ", id(a))
b = a.copy()
print("id of b: ", id(b))
b.shape = 3,3
b[0][0] = 10
print("reshaped b: \n", b)
print("new a: \n", a)

# TODO: NumPy - Matrix Library and how it is different from ndarray stuff