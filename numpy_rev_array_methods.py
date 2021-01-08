import numpy as np

# In-buold numpy array manipulation methods

######## Changing Shapes #########

'''

# reshape: Gives a new shape to an array without changing its data
# Syantax: numpy.reshape(arr, newshape, order')
# arr -> source array
# newshape -> tuple of the shape. Note it has to be compatilble with original shape
# order -> C-style (row-wise), F-stype (column-wise)

a = np.arange(24)
b = a.reshape(3,8)
c = a.reshape(8,3)
d = a.reshape(2,3,4)
print("Reshaped array: ", a)
print("Reshaped array: ", b)
print("Reshaped array: ", c)
print("Reshaped array: ", d)
'''

'''
# flaten: returns a copy of an array collapsed into one dimension
# Syantax: ndarray.flatten(order)
# order: C-style (row-wise), F-stype (column-wise)

a = np.arange(10).reshape(2,5)
print("Original array: ", a)
b = a.flatten()
print("Flattened 1-array: ", b)
'''

###### Transpose ops #####
'''
# traspose: permutes the dimension of the given array. It returns a view wherever possible.
# Syantax: numpy.transpose(arr, axes)
# arr -> numpy array to be transposed
# axes -> List of ints, corresponding to the dimensions. By default, the dimensions are reversed

a = np.arange(24).reshape(2, 3, 4)
print("Original array: \n", a, " \nshape: ", a.shape)
b = np.transpose(a)
print("Transposed array: \n ", b, " \nshape: ", b.shape)
'''

# Another shoter way to perform transpose is arr.T

'''
# swapaxes: interchanges the two axes of an array
# Syantax: numpy.swapaxes(arr, axis1, axis2)

a = np.arange(8).reshape(2,2,2) 
print("Original array: \n", a)
print("Swap numbers between axis 2 and axis 0 \n", np.swapaxes(a, 0, 2))

# TODO: a more complex swaping can be done using rollaxis funtion
'''

##### Changing Dimensions or Boradcasting ######

# broadcast: Produces an object that mimics broadcasting
# Syantax: a.broadcast(arr,1 arr2)

'''
x = np.array([[1], [2], [3]]) 
y = np.array([4, 5, 6])  
   
# tobroadcast x against y 
b = np.broadcast(x,y)  
print("braodcast array:  \n ")
for (u,v) in b:
    print(u, ",", v)
'''
'''
# braodcast_to: broadcasts an array to a new shape. It returns a read-only view on the original array
# Syantax: numpy.broadcast_to(array, shape, subok)

a = np.arange(4).reshape(1,4)
print('The original array: \n', a)
print("Broadcased array: \n", np.broadcast_to(a, (4,4)))  
'''

'''
# expand_dims: This function expands the array by inserting a new axis at the specified position
# Syantax: numpy.expand_dims(arr, axis)
# axis -> Position where new axis to be inserted

x = np.array( [ [1,2], [3,4], [5,6] ] )
print("Original array: \n", x, " shape: ", x.shape)
y = np.expand_dims(x, axis = 0)
print("expanded array: \n", y, " Shape: ", y.shape)
y = np.expand_dims(x, axis = 1)
print("expanded array: \n", y, " Shape: ", y.shape)
'''

'''
# squeeze: This function removes one-dimensional entry from the shape of the given array
# Syantax: numpy.squeeze(arr, axis)
# axis -> int or tuple of int. selects a subset of single dimensional entries in the shape

x = np.arange(9).reshape(1,3,3) 

print ('Array X: \n',x) 
y = np.squeeze(x) 

print ('Array Y: \n', y) 

print ('The shapes of X and Y array:' )
print (x.shape, y.shape)
'''


###### Joining arrays ######
'''
# concatenate: This function is used to join two or more arrays of the same shape along a specified axis.
# Syantax: numpy.concatenate((a1, a2, ...), axis)
# axiz-> Axis along which arrays have to be joined. Default is 0

a = np.array( [ [1,2],[3,4] ] )
b = np.array( [ [6,7],[8,9] ] )
c = np.concatenate((a,b), axis = 1)
print("Concatenated array: \n", c, " shape: ", c.shape)
c = np.concatenate((a,b))
print("Concatenated array: \n", c, " shape: ", c.shape)
'''

'''
# stack: This function joins the sequence of arrays along a new axis
# Syantax: numpy.stack(arrays, axis)
# arrays -> Sequence of arrays of the same shape
# axis -> Axis in the resultant array along which the input arrays are stacked

a = np.array( [ [[1,2],[3,4], [5,6] ]] )
b = np.array( [ [[6,7],[8,9], [10,11] ]] )
b1 = np.array( [ [[6,7],[8,9], [10,11] ]] )
print("Shape of a & b: ", a.shape, " ", b.shape)
c = np.stack((a,b, b1), 0)
print("Stacked array along with axis 0: \n", c, " shape: ", c.shape)
c = np.stack((a,b, b1), 1)
print("Stacked array along with axis 1: \n", c, " shape: ", c.shape)

# Note: so the difference between the concatenate and stack is in stacking the given axis is always summed up
# Note: similar to stack there is hstack and vstack functions as well
'''

##### Splitting Array ######


# split: divides the array into subarrays along a specified axis
# Syantax: numpy.split(ary, indices_or_sections, axis)
# indices_or_sections ->
# Can be an integer, indicating the number of equal sized subarrays to be created from the input array. 
# If this parameter is a 1-D array, the entries indicate the points at which a new subarray is to be created.

'''
# 1-D Array split
a = np.arange(9)
print("split arrray equally: \n", np.split(a, 3))
print("Split array at given location: \n", np.split(a, [4,7]))

# 2-D array split
a = np.arange(9).reshape(3,3)
print("split arrray equally: \n", np.split(a, 3))
print("Split array at given location: \n", np.split(a, [1,3]))
'''

'''
# hsplit: The numpy.hsplit is a special case of split() function where axis is 1 indicating a horizontal split regardless of the dimension of the input array.
# vsplit: numpy.vsplit is a special case of split() function where axis is 1 indicating a vertical split regardless of the dimension of the input array
a = np.arange(16).reshape(4,4) 
print("Original array: \n", a)

b = np.hsplit(a,2) 
print ('Horizontal splitting: \n', b) 

b = np.vsplit(a, 2) 
print ('Vertical splitting: \n', b) 
'''

###### Element & Description ######

'''
# resize: This function returns a new array with the specified size. 
# If the new size is greater than the original, the repeated copies of entries in the original are contained.
# Syantax: numpy.resize(arr, shape)

a = np.array([[1,2,3],[4,5,6]])
print("Origianl array: \n", a, "Shape: ", a.shape)
b = np.resize(a, (3,2))
print("Resize with same size: \n", b, "Shape: ", b.shape)

b = np.resize(a, (4,4, 2))
print("Resize with greter size: \n", b, "Shape: ", b.shape)
'''

# append: This function adds values at the end of an input array. The append operation is not inplace, a new array is allocated. 
# Also the dimensions of the input arrays must match otherwise ValueError will be generated.
# Syantax: numpy.append(arr, values, axis)
# values -> To be appended to arr. It must be of the same shape as of arr (excluding axis of appending)
# axis -> The axis along which append operation is to be done. If not given, both parameters are flattened

a = np.array([[1,2,3],[4,5,6]])
print("Orignial array: \n", a)
print ("append: \n", np.append(a, [7,8,9]))
print ("append axis 0: \n", np.append(a, [[7,8,9]],axis = 0))
print ("append axis 1: \n", np.append(a, [[0,0,0],[7,8,9]],axis = 1))