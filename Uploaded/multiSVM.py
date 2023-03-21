import numpy as np
import matplotlib
import copy
import random
from utils import mnist_reader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load F-MNIST data
X_train, Y_train = mnist_reader.load_mnist('./data', kind='train')
X_test, Y_test = mnist_reader.load_mnist('./data', kind='t10k')


#==============================================================#
# YOUR CODE HERE
# Play around with different values of the regression parameter

reg_param = 0.1

#==============================================================#
#==============================================================#

clf = make_pipeline(StandardScaler(), SVC(C=reg_param))

clf.fit(X_train, Y_train)

print ('Test Accuracy:',np.mean(Y_test==clf.predict(X_test)))