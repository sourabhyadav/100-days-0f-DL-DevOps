# Numpy revision for some basic to advance concepts 

import numpy as np

# Make a simple array
arr = np.array([1,2,3])

# Multi dime array
arr_nd = np.array([ [ [ [1,2,2], [3,4,3], [1,2,3] ], [ [5,6,4], [7,8,7], [1,2,3] ], [ [1,2,3], [1,2,3],[1,2,3]  ] ], [ [ [1,2,2], [3,4,3], [1,2,3] ], [ [5,6,4], [7,8,7], [1,2,3] ], [ [1,2,3], [1,2,3],[1,2,3]  ] ]  ])
print(arr_nd.shape)

# We can create user defiend structured data types. Each stuff acts like a column similar to pandas
student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
record = np.array([ ('sourabh', 32, 80), ('shubham', 24, 50) ], dtype = student )
print(record, record.shape, record[0])

 # Resize the numpy arrays
a = np.array([ [1,2,3], [4,5,6] ])
print("Orig shape:", a.shape)
a.shape = (3,2)                     # Resized shape has to be multiple of the rowxcol
print("Resized shape: ", a.shape)
new_a = a.reshape(1,6)              # Resize the array with reshape function. Always use this method
print("new shape: ", new_a.shape)

# Initi an array 
a = np.arange(24)
print("Array is: ", a, " dim: ", a.shape)
b = a.reshape(3,4,2)
print("New array shape: ", b, " dim: ", b.shape)

# The itemsize attribute
x = np.array([1,2,3,4,5], dtype = np.float32) 
print (x.itemsize)
