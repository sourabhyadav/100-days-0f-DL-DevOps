import numpy as np 

a, b = 13, 15
print ("AND ops: ", np.bitwise_and(13, 17))
print("OR ops: ", np.bitwise_or(13, 17))
print("Invert of 13 :", np.invert(np.array([13], dtype = np.uint8)))
print("Left shift of 10 by 2 position ", np.left_shift(10,2) )
print("Right shift of 40 by 2 position ", np.right_shift(40,2))

# get the binary rep
a = bin(a)
b = bin(b)
print("bin(a): ", a, " bin(b): ", b)

