import numpy as np

# Idea is to selecting arbitary items from the given numpy array
# 
x = np.array([[1, 2], [3, 4], [5, 6]]) 
y = x[[0,1,2], [0,1,0]]                 # Selection includes the elements of (0,0), (1,1), (2,0)
print ("Random indexex array is: ",y)

# Select elements of corners of 4x3 array
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
y = x[ [[0,0], [3,3]], [[0,2], [0,2]] ]     # Basically for each dimension a combination 1st dim and 2nd dim
print("Corner of 4x3 matrix is", y)

# Boolean Indexing
x = np.array([[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]]) 
y = x[x>4]
print("All the elementes which are greater than 4: ", y)

