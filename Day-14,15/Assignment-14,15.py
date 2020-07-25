import numpy as np


# Q1 Create a 3x3x3 array with random values
darray = np.arange(27).reshape(3,3,3)
print(darray,type(darray))


# Q2 Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
q2 = np.diag(1+np.arange(4), k = -1) 
print (q2,type(q2),q2.shape)
# numpy.diag(v k)
#v is a 2-D array, return a copy of its k-th diagonal. If v is a 1-D array, return a 2-D array with v on the k-th diagonal.
#k =  Diagonal in question. The default is 0. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal


# Q3 Create a 8x8 matrix and fill it with a checkerboard pattern
q3 = np.zeros ((8,8), dtype=int)
print (q3,type(q3),q3.shape)
q3[1::2, ::2]= 1 # Acessing Index Range with a step of 2
q3[::2, 1::2] = 1 # Second Update for the other alternate rows
print (q3,type(q3),q3.shape)

# Q4 Normalize a 5x5 random matrix
q4 =np.random.random_sample((5,5))# random array of size 5,5
norm=np.linalg.norm(q4) 
normal_array = q4/norm
print(normal_array)
#q4max, q4min = q4.max(),q4.min()
#q4=(q4-q4min)/(q4max-q4min)
#print (q4,type(q4),q4.shape)


# Q5.How to find common values between two arrays?
q51 = np.random.randint(0,10,10)
q52 = np.random.randint(0,10,10)
print(q51,q52)
print(np.intersect1d(q51,q52))
#numpy.intersect1d(ar1, ar2, assume_unique=False, return_indices=False)  ---Find the intersection of two arrays.Return the sorted, unique values that are in both of the input arrays.

# Q6. How to get the dates of yesterday, today and tomorrow?
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today = np.datetime64('today', 'D')
tomorrow = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print ("Yesterday was : ",yesterday, "\n Today is :",today,"\n Tomorrow will be :",tomorrow)


# Q7. Consider two random array A and B, check if they are equal
A = np.random.randint(0,10,10)
B = np.random.randint(0,10,10)
print("Array A is :",A,"\n Array B is :",B)
check = np.allclose(A,B)
if check :
    print("The random Arrays are equal")
else:
    print("The random Arrays are not Equal")


# Q8. Create random vector of size 10 and replace the maximum value by 0
q8 =  np.random.random(10) 
print("The Vector is  : \n", q8)
m =np.amax(q8) #finding max value
print("\n Max element : ",m )
q8[q8.argmax()]=0 #Returns the indices of the maximum values along an axis and replace it with 0
print("\n The Vector after replacement of max Value is  :",q8)

# Q9. How to print all the values of an array?
#https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html?highlight=set_printoptions#numpy.set_printoptions

# Q10.Subtract the mean of each row of a matrix
q10 = np.random.rand(2, 2) # Random Matrix
print("Original Random Matrix : \n",q10)
q101 = q10 - q10.mean(axis=1, keepdims=True) # If this is set to True, the axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array.
q101 = q10 - q10.mean(axis=1).reshape(-1, 1)
print("\n After Substraction of Mean of each row :\n",q101)

# Q11. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?
Z = np.ones(10) # Ones Vector
print(Z)
I = np.random.randint(0,len(Z),20) # Random having length of original Vector,  numpy.random.randint(low, high=None, size=None, dtype=int)¶
print(I)
Z += np.bincount(I, minlength=len(Z)) # bincount = Count number of occurrences of each value in array of non-negative ints.
print(Z)


# Q12. How to get the diagonal of a dot product?
A = np.random.randint(0,10,(3,3))
B= np.random.randint(0,10,(3,3))
c1 = np.dot(A,B)
print("Random Matrix 1 :\n",A,"\n\n Random Matrix 2 :\n",B,"\n\n Dot Product of the Two Matrices :\n",c1)
d1=np.sum(A * B.T, axis=1)
print("\n Diagonal of the Matrix :\n",d1)

# Q13. How to find the most frequent value in an array?
Z = np.random.randint(0,5,10)
print (Z)
print('Most Frequented Value:', np.bincount(Z).argmax()) #bincount


#Q14. How to get the n largest values of an array
Z = np.arange(10)
print(Z) # Original Array
np.random.shuffle(Z)
print(Z) # After Shuffling
n = 2 # n largest values from the array
print (Z[np.argsort(Z)[-n:]]) #sorting and printing

# Q15. How to create a record array from a regular array?
x1=np.array([1,2,3,4])
x2=np.array(['a','dd','xyz','12'])
x3=np.array([1.1,2,3,4])
r = np.core.records.fromarrays([x1,x2,x3],names='a,b,c')
print(r)


