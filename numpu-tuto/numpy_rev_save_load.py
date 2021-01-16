import numpy as np
# NumPy introduces a simple file format for ndarray objects. This .npy file stores data, shape, dtype and other information required to reconstruct the ndarray in a disk file such that the array is correctly retrieved even if the file is on another machine with different architecture.

a = np.array([1,2,3,4,5]) 
np.save('outfile',a)

b = np.load('outfile.npy') 
print ("Loaded .npy b: ", b) 

a = np.array([1,2,3,4,5]) 
np.savetxt('out.txt',a) 
b = np.loadtxt('out.txt') 
print ("Loaded .txt b: ", b)