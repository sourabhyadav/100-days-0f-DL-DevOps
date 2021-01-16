import numpy as np

# Element-wise numltiplication
a = np.array( [1,2,3,4 ] )
b = np.array( [10,20,30,40 ] )
print("Element-wise multiplicaiton: ", a*b)

######### Broadcasting Rules #################
# https://www.tutorialspoint.com/numpy/numpy_broadcasting.htm

a = np.array([[2.0,2.0,2.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])
# Braodcasting works differently in both the addition and multiplication cases check the above link
print("Broadcasting addtion: ", a + b)          
print("Broadcasting multipilcation: ", a*b)

# TODO: More broadcasting and Arthmetic expamles required