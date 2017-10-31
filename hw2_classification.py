from __future__ import division
import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2])
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

## can make more functions if required

    
def pluginClassifier(X_train, y_train, X_test):    
  # this function returns the required output 
  class_ = [x for x in range(10)]
  n = y_train.shape[0]
  mu_list = []
  sig_list = []
  pi_list = []
  for y in class_:
      n_y = np.sum(y_train == y)
      pi_hat = 1/n*np.sum(y_train == y)
      pi_list.append(pi_hat)
      mu_hat = 1/n_y*np.sum(X_train*np.expand_dims((y_train == y), -1), axis = 0)
      mu_list.append(mu_hat)
#      sum_ = 0
#      for i in range(n):
#          x_i = X_train[i]
#          y_i = y_train[i]
#          s = (y == y_i)*np.outer((x_i-mu_hat),(x_i - mu_hat))
#          sum_ += s
#      sigma_hat = (1/n_y)*sum_
      
      #use np.cov to calculate the covariance
      sap = X_train[y_train == y,:]
      sigma_hat = np.cov(sap.T)
      sig_list.append(sigma_hat)    
  out = []

  for i in range(X_test.shape[0]):
      x = np.expand_dims(X_test[i] , -1)
     
      y_class = []
      for ii in range(10):
          expo = -1/2*np.dot(np.dot((x.T-mu_list[ii]),np.linalg.inv(sig_list[ii])), (x.T-mu_list[ii]).T)
          y = pi_list[ii]*np.linalg.det(sig_list[ii])**(-1/2)*np.exp(expo)
          y_class.append(y)
      out.append(y_class)
  normed = []
  for x in out:
      m = np.sum(x)
      x = x/m
      normed.append(x)
    
  return normed
       
      

final_outputs = pluginClassifier(X_train, y_train, X_test) # assuming final_outputs is returned from function

np.savetxt("probs_test.csv", final_outputs, delimiter=",") # write output to file