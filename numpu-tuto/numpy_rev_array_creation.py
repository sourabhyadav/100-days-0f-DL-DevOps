import numpy as np

# Array Creation methods
a = np.empty( (2,3,4), dtype= int )     # Randomly UNinitilizes the array with random integer
a_zeros = np.zeros((2,3,1), dtype= int) # All initilizes to zero
a_zeros = np.zeros((2,3,1), dtype= 'i4') # All initilizes to zero float
x = np.zeros((2,2), dtype = [('x', 'i4'), ('y', 'i4')])     # Initilize with other datatype
x = np.ones([2,2], dtype = int)             # Similar to zeros


# Converting any python datatype to numpy arrays
# numpy.asarray(a, dtype = None, order = None)
# Where a --> Input data in any form such as list, list of tuples, tuples, tuple of tuples or tuple of lists

# Convert from a list of numbers
x = [ [1,2,3], [4,5,6] ] 
a = np.asarray(x, dtype = float)
print(a)

# Convert from a array of tuples
x = [(1,2,3),(4,5)] 
a = np.asarray(x) 
print(a)

# Numpy range functions
# Syntax: numpy.arange(start, stop, step, dtype)

a = np.arange(2, 20, 3)
print("Array from range: ", a)

# Linear space function
# Syantax: numpy.linspace(start, stop, num, endpoint, retstep, dtype)
# num --> number of entries to take
# endpoint --> boolean to include the end number or not, default = True
x = np.linspace(10,20, 5, endpoint = False)
print("Linear array is: ", x)

# Log space function
# Synatax: numpy.logspace(start, stop, num, endpoint, base, dtype)
# Similar to linera space
a = np.logspace(1,10,num = 10, base = 2)
print("Log space array: ", a)

