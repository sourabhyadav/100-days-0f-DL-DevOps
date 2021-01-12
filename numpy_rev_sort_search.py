import numpy as np

'''
# sort
# Syantax: numpy.sort(a, axis, kind, order)

a = np.array([[3,7],[9,1]]) 
print("sort default: ", np.sort(a))
print("sort axis = 0: ", np.sort(a, axis = 0))

# VIP sorting based on custom data type
dt = np.dtype([('name', 'S10'),('age', int)]) 
a = np.array([("raju",21),("anil",25),("ravi", 17), ("amar",27)], dtype = dt) 
print("print with a given datatype \n", np.sort(a, order= 'age'))

# argsort
x = np.array([3, 1, 2])
y = np.argsort(x) 
print("input array: ", x)
print("indices for sorting: ", y)
print("reconstruct the arry: ", x[y])
'''

# TODO: lexsort

'''
# argmx, argmin
a = np.array([[30,40,70],[80,20,10],[50,90,60]]) 
print("input array: ", a)
print ("argmax: ", np.argmax(a))
print ("argmax axis=0: ", np.argmax(a, axis =0))
minindex = np.argmin(a)
print ("with minindex ", a.flatten()[minindex]) 
'''

# where: returns indices where it mataches the condition
a = np.array([[30,40,0],[0,20,10],[50,0,60]])  
y = np.where(a > 30)
print("indices are: ", y)
print("get the array with where: ", a[y])

# extract: The extract() function returns the elements satisfying any condition.
x = np.arange(9.).reshape(3, 3) 
print ("return the values where condition is correct: ", np.extract(np.mod(x, 2) == 0, x))