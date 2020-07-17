import numpy as np

X1= np.array([[0,1, 2]])
X2= np.array([[2,1,0]])

X = np.stack((X1, X2), axis=0)
print(np.cov(X1,X2))