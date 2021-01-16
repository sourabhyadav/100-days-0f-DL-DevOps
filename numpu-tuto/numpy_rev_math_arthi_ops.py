import numpy as np

# Mathematical functions
a = np.array([0,30,45,60,90]) 
# Note: all the functions take radian as angle

# sin
print ("print sin: ", np.sin(a*np.pi/180))

# cos
print ("print cos: ", np.cos(a*np.pi/180))

# tan
print ("print tan: ", np.tan(a*np.pi/180))

# arcsin
sin = np.sin(a*np.pi/180)
inv = np.arcsin(sin)
print("inv sin: ", np.degrees(inv))

# arccos
cos = np.cos(a*np.pi/180)
inv = np.arccos(cos)
print("inv sin: ", np.degrees(inv))

# arctan
tan = np.tan(a*np.pi/180)
inv = np.arctan(tan)
print("inv sin: ", np.degrees(inv))


# Rounding ops
# Syantax: 
#numpy.around(a,decimals)
# decimals -> The number of decimals to round to. Default is 0. If negative, the integer is rounded to position to the left of the decimal point

a = np.array([1.0,5.55, 123, 0.567, 25.532]) 
print("Round: ", np.round(a))
print("Round deci = 1 ", np.round(a, decimals  = 1))
print("Round deci = -1 ", np.round(a, decimals = -1))

# floor: This function returns the largest integer not greater than the input parameter. 
a = np.array([-1.7, 1.5, -0.2, 0.6, 10]) 
print("Floor ops: ", np.floor(a))

# ceil: The ceil() function returns the ceiling of an input value, i.e. the ceil of the scalar x is the smallest integer i, such that i >= x
print("Ceil ops: ", np.ceil(a))

# add, subtract, multiple, divide 
# Note: all the ops follow broadcasting and element-wise rules
a = np.arange(9, dtype = np.float_).reshape(3,3) 
b = np.array([10,10,10]) 

print("a: \n", a)
print("b: \n", b)
print("add array: ", np.add(a,b))
print("subtract array: ", np.subtract(a,b))
print("add multiply: ", np.multiply(a,b))
print("add divide: ", np.divide(a,b))

# power
a = np.array([10,100,1000])
print("power with scalar: ", np.power(a, 2))
b = np.array([1,2, 3])
print("power with array elementwise: ", np.power(a, b))

# mod or remainder
a = np.array([10,20,30]) 
b = np.array([3,5,7]) 

print("mod with scalar: ", np.mod(1, 4))
print("mod with array elementwise: ", np.mod(a,b))
print("remainder with elementwise: ", np.remainder(a,b))