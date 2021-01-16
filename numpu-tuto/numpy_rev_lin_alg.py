import numpy as np

# dot
# For 2-D vectors, it is the equivalent to matrix multiplication.
# For 1-D arrays, it is the inner product of the vectors
# For N-dimensional arrays, it is a sum product over the last axis of a and the second-last axis of b

a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
c = np.dot(a,b) # [[1*11+2*13, 1*12+2*14],[3*11+4*13, 3*12+4*14]]
print("dot prod: ", c)

# vdot
a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
print ("vdot: ", np.vdot(a,b)) # 1*11 + 2*12 + 3*13 + 4*14 = 130

# inner
print ("inner product of 2 1-d vectors: ", inner prodc""np.inner(np.array([1,2,3]),np.array([0,1,0])) 
# Equates to 1*0+2*1+3*0

a = np.array([[1,2], [3,4]])
b = np.array([[11, 12], [13, 14]]) 
print("inner produc: ", np.inner(a,b)) 
# [[1*11+2*12, 1*13+2*14] 
# [3*11+4*12, 3*13+4*14]]  

# matmul
a = [[1,0],[0,1]] 
b = [[4,1],[2,2]] 
print ("Matmul: ", np.matmul(a,b))

a = [[1,0],[0,1]] 
b = [1,2] 
print ("matmul array n vector: ", np.matmul(a,b)) 
print ("matmul vector n array: ", np.matmul(b,a))

a = np.arange(8).reshape(2,2,2) 
b = np.arange(4).reshape(2,2) 
print ("matmul with broadcast: ", np.matmul(a,b))

# TODO: work on determinant, inv and solve methods in numpy