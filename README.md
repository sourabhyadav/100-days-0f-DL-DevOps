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