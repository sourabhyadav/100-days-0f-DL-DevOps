import numpy as np 
from matplotlib import pyplot as plt 

x = np.arange(1,11) 
y = 2 * x + 5 
plt.title("Matplotlib demo") 
plt.xlabel("x axis caption") 
plt.ylabel("y axis caption") 
plt.plot(x,y,"ob") 
plt.show() 

x = np.arange(0, 3 * np.pi, 0.1) 
y_sin = np.sin(x) 
y_cos = np.cos(x)  
y_tan = np.tan(x)
   
# Set up a subplot grid that has height 2 and width 1, 
# and set the first such subplot as active. 
plt.subplot(2, 2, 1)    # last index show the which subplot is active
   
# Make the first plot 
plt.plot(x, y_sin) 
plt.title('Sine')  
   
# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 2, 2) 
plt.plot(x, y_cos) 
plt.title('Cosine')  

# Set the second subplot as active, and make the second plot. 
plt.subplot(2, 2, 3) 
plt.plot(x, y_tan) 
plt.title('Tan')

x = [5,8,10] 
y = [12,16,6]  

x2 = [6,9,11] 
y2 = [6,15,7] 

plt.subplot(2, 2, 4)
plt.bar(x, y, align = 'center') 
plt.bar(x2, y2, color = 'g', align = 'center') 
plt.title('Bar graph') 
plt.ylabel('Y axis') 
plt.xlabel('X axis') 

# Show the figure. 
plt.show()

a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27]) 
plt.hist(a, bins = [0,20,40,60,80,100]) 
plt.title("histogram") 
plt.show()