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

