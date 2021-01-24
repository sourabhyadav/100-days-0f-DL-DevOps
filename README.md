# 100-days-0f-DL-DevOps
Summary of my learning in 100 days of DL and DevOps related stuff. 

### Day 1: Numpy Revision 1  
* Revised few concepts of ndarray and methods related to init and built-in functions
* Started with concepts required in deep learning wrt to numpy  

### Day 2: Numpy Revision 2  
* Revised the concept of indexing and slicing (:) and elipcies (...) in numpy
* Slicing works on each dimesion of the given array. 
* Basic syntax of slicing for each dimesion is start:stop:step

### Day 3: Array Broadcasting and Iteration  
* Broadcasting has some rules which performs broadcasting for addition and multiplication differently  
* We can also use broadcasting based on boolean condtion inside the array   
* For iteration we can use nditer function and can also define the C-style(row-wise) and F-style(col-wise)   

### Day 4: Array manipulation using built-in functions 1  
* Methods for chaning shape like reshape, flaten 
* Methods for transpose like tranpose(), a.T, swapazes etc.   
* Methods for changing dimension and broadcasting like broadcast, broadcast_to etc.

### Day 5: Array manipulation using built-in functions 2  
* Methods for chaning contenate and split  
* Methods for elements and deleting them   

### Day 6: Array binary and string operations   
* Methods for binary ops like AND, OR, Shift left/right  
* Methods for string related operations mostly they are similar to python string ops 

### Day 7: Mathematical, arthematic, statiscal, sorting and searching methods   
* Methods for mathematical ops like sin, cos, tan , arcsing, etc.  
* Methods for arthimatic ops like add, subtract, multiply, divide, round, floor, ceil
* Methods for statistical like mean, median, variance, weighted avg, percentile etc.
* Methods for sorting and searching like sort, argsort, lexsort, argmax, argmin, where, extract etc. 

### Day 8: Numpy shallow n deep copy, linear algebra and matplotlib, save and load as .npy    
* Methods for no copy: '=', for shallow copy: view() and slicing, deep copy: copy().  
* Methods for linear algebra are dot, cross, inner, etc. 
* Methods for plotting the graphs in 2-d or 3-d also understood the subgraph concept 
* Methods for loading and saving ndarrays. By default it saves in .npy format which we cannot read. There is another method to store the nd-array in savetxt and loadtxt methods which are human readable.   

### Day 9: Pytorch basic tensor  
* Methods related to basic maths in tensor matircs, slicing and broadcasting
* Tensor conversion from CPU to GPU and GPU ops etc.
* Converting numpy array to pytorch tensor and vice-versa

### Day 10: Pytorch backpropagation and autograd utility  
* Learnt about auto_grad utility to automatically calulate gradients and track/untrack gradients 
* Apply a dummy training with 4-setp process 1. forward pass(compute loss), 2. Calculate grad(partial derivatives), 3. Backward pass (apply chain rule), 4. Update weights(gradient decent process), 5. Apply optimizer (to update the weights)

### Day 11: Pytorch based training with pytorch loss and optimizer  
* Learnt about how to use pytorch loss and optimizer calss
* Also understood how to initilize our on layers 
* This is one the the most important lecture of the pytorch training

### Day 12: Pytorch based Linear and Logistic Regression  
* Learnt about different loss functions for both linear and logistic regrassion
* Learnt about using datasets from sklearn to use for basic machine learning application

