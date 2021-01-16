import numpy as np
'''
# min-max
a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 
print("input array: \n", a)
print("min: ", np.amin(a))
print("min axis =0 ", np.amin(a, axis=0))
print("min axis =1 ", np.amin(a, axis=1))
print("max: ", np.amax(a))
print("max axis =0 ", np.amax(a, axis=0))
print("max axis =1 ", np.amax(a, axis=1))
'''
# TODO: tpt and percentile methods
'''
# median & mean
a = np.array([[30,65,70],[80,95,10],[50,90,60]])
print("input array: \n", a)
print("median: ", np.median(a))
print("median axis =0 ", np.median(a, axis=0))
print("median axis =1 ", np.median(a, axis=1))
print("mean: ", np.mean(a))
print("mean axis =0 ", np.mean(a, axis=0))
print("mean axis =1 ", np.mean(a, axis=1))
'''
'''
# weighted average

a = np.array([1,2,3,4])
wts = np.array([4,3,2,1]) 
print ("weighted avg: ", np.average(a,weights = wts))
'''

# std deviation
# Formula: std = sqrt(mean(abs(x - x.mean())**2))
print ("standard deviation: ", np.std([1,2,3,4]))

# variance
# Formula mean(abs(x - x.mean())**2)
print("variance: ", np.var([1,2,3,4]))