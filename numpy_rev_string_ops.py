import numpy as np 

# add
print("Aded strings: ", np.char.add(['hellow'], ['xtz']))
print("nd string add: Note the addtion format: ", np.char.add(['hello', 'hi'], ['abc', 'xyz']))

# multiply
print ("multiply string 3 times :",  np.char.multiply('Hello ',3))

# center
print ("put my string array in the center: ",  np.char.center('hello', 20,fillchar = '*'))

# capitalize
print ("small to captilize ", np.char.capitalize('hello world'))

# upper
print ("lower to upper: ", np.char.upper('hello')) 
print ("nd lower to upper: ", np.char.upper(['hello','world']))

# lower
print ("nd upper to lower: ", np.char.lower(['HELLO','WORLD']))

# title
print ("titize: ", np.char.title('hello how are you?'))

# split
print ("split by space", np.char.split ('hello how are you?') )
print ("splist by ,: ", np.char.split ('TutorialsPoint,Hyderabad,Telangana', sep = ','))

# splitlines
print ("splitlines: ", np.char.splitlines('hello\nhow are you?')) 
print ("splitlines: ", np.char.splitlines('hello\rhow are you?'))

# join VIP
print ("join 1d: ", np.char.join(':','dmy') )
print ("join nx: ", np.char.join([':','-'],['dmy','ymd']))

# strip
print ("strip 1-d:", np.char.strip('ashok arora','a')) 
print ("strip n-d: ",np.char.strip(['arora','admin','java'],'a'))

# replace
print ("replace is to was: ", np.char.replace ('He is a good boy', 'is', 'was'))

# encode and decode
a = np.char.encode('hello', 'cp500') 
print ("encoded :" ,a) 
print ("decoded: ", np.char.decode(a,'cp500'))
