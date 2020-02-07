import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets

### 1 GENERATE DATA
iris = datasets.load_iris()
### Pay attention that "X" is a (150, 4) shape matrix
### y is a (150,) shape array
X = iris.data
y = iris.target


### 2 CENTER DATA
### "X" has a (150,4) shape, so we need define what kind of mean do we want
### you need to use "np.mean()" function
### axis=0     : gives us (4,) 
### axis=1     : gives us shape (150,)
### axis=None  : gives us a shape () <-- this is a number
X_centered = X - X.mean(axis=0)
X_centered = X_centered.T

### 3 PROJECT DATA
### at first you need to get covariance matrix
### Pay attention that cov_mat should be a (4, 4) shape matrix
cov_mat = np.cov(X_centered)
### next step you need to find eigenvalues and eigenvectors of covariance matrix
eig_values, eig_vectors = np.linalg.eig(cov_mat)
### if you run this code
print( eig_values / eig_values.sum() )
### you can choose the 2 eigenvectors with the highest values
index_1 = 0
index_2 = 1
print(f"this is our 2D subspace:\n {eig_vectors[:, [index_1,index_2]]}")
### now we can project our data to this 2D subspace
### project original data on chosen eigenvectors
### (4,2).T  *  (4,150)  =  (2,150) <-- projected data
projected_data  = np.dot(eig_vectors[:, [index_1, index_2]].T, X_centered)
### now you are able to visualize projected data
### you should get excactly the same picture as in the last lab slide
plt.plot(projected_data[0, y == 0], -projected_data[1, y == 0], 'bo', label='Setosa')
plt.plot(projected_data[0, y == 1], -projected_data[1, y == 1], 'go', label='Versicolour')
plt.plot(projected_data[0, y == 2], -projected_data[1, y == 2], 'ro', label='Virginica')
plt.show()

### 4 RESTORE DATA
### we have a "projected_data_local" which shape is (2,150)
### and we have a 2D subspace "eig_vectors[:, [index_1, index_2]]" which shape is (4,2)
### how to recieve a restored data with shape (4,150)?
### (4,2)  *  (2,150)  =  (4,150) <-- restored data
### and now we need to shift data back with help of a vector of mean
### (4, 150) + (4,)*np.ones(shape=(4,150)) = (4, 150) <-- broadcast operation
restored_data = np.dot(eig_vectors[:, [index_1, index_2]], projected_data)
restored_data = restored_data + X.mean(axis=0, keepdims=True).T


############################################
### CONGRATS YOU ARE DONE WITH THE FIRST PART ###
############################################

### 1 GENERATE DATA
### already is done

### 2 CENTER DATA
### already is done

### 3 PROJECT DATA
### "n_components" show how many dimensions should we project our data on 
pca = decomposition.PCA(n_components=2)
### class method "fit" for our centered data
pca.fit(X_centered.T)
### make a projection
X_pca = pca.transform(X_centered.T)
### now we can plot our data and compare with what should we get
plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)
plt.show()











