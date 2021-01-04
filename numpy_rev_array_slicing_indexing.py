import numpy as np

# Python based slicing
a = np.arange(10)
s = slice(2, 7, 2)      # Python slice object. Syntax slice(start, stop, step)
b = a[s]                # Wow. The slice object can be direcly used to numpy array
print("Original array: ", a, " Python slicing: ", b)

print("################# Slicing y Index #####################")

# Slicing by indexing VIP
# Syntax: for each dimesion in the array we need to follow start:stop:step syntax

print("#### 1-d array slicing ####")

# 1-d array slicing
a = np.arange(24)
print("slicled: ", a[2:10:3])           # start=2:stop=10:setop=3
print("sliced before: ", a[2:])         # this can be read as 2:end:1
print("sliced after: ", a[:6])           # this can be read as 0:6:1
print("sliced before with step: ", a[:10:2])    # this can be read as 0:10:2

print("#### n-d array slicing ####")

# n-d array slicing with elipsies ...
a = np.array([[1,2,3],[4,5,6],[7,8,9]]) 
print("Original array: ", a)

# this returns array of items in the second column 
print ('The items in the second column are:')  
print (a[...,1])        # row= all, col= 1 i.e all the rows and column index 1

# Now we will slice all items from the second row 
print ('The items in the second row are:' )
print (a[1,...])          # row= 1, col= all i.e. row index 1 and all the collumns

# slice items starting from index
print ('Now we will slice the array from the index a[1:]') 
print (a[1:])                 # this can be read as in the first dim which is row from [1:end:1, ...]

# Now we will slice all items from column 1 onwards 
print ('The items column 1 onwards are:') 
print (a[...,1:])             # this can be read as [all rows (1st dime), columns from 1:end:1]

# Indexing synatax applies to each dim of the given array
print("Upper 2x2 matrix: ", a[:2, :2])          # this can be read as [0:2:1, 0:2:1]
print("Lower 2x2 matrix: ", a[1:, 1:])          # this can be read as [1:end:1, 1:end:1]
print("Read only last column: ", a[...,2])      # this can be read as [0:end:1, 2]