# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:38:18 2017

@author: esppk
"""

import numpy as np
from scipy.stats import multivariate_normal as Normal
import sys
file_name = str(sys.argv[1])
 

#%%
X = np.random.randn(100, 4)

#%%

#K-means

k = 5
n_obs = X.shape[0]
mu_idx = np.random.choice([x for x in range(n_obs)], k)
mu = [X[i, :] for i in mu_idx]

c = [int(c_i) for c_i in np.zeros(n_obs)]

for n in range(10):
    for i in range(X.shape[0]):
        
        x_i = X[i, :]
        dist = list()
        for ii in range(k):
            mu_k = mu[ii]
            dist.append(np.sum((x_i - mu_k)**2))
            
        d = np.array(dist)
        c[i] = d.argmin()
        

    
    for kk in range(k):
        c_1 = (np.array(c) == kk)
        c_filter = np.expand_dims(c_1, -1)
        n_k = np.sum(c_1) 
        mu[kk] = 1/n_k*np.sum(X*c_filter, axis = 0)
    
#    np.save("centroids-{}.csv".format(n), mu)
    file = "centroids-{}.csv".format(n+1)
    np.savetxt(file, mu, delimiter=",")
    

#EM GMM
#setting cov matrix
d =  X.shape[1]

Sigma = np.eye(d)

#initialize pi
pi = [0.2]*5
pi = np.array(pi)
#initialize mu
mu_idx = np.random.choice([x for x in range(100)], k)
mu = [X[i, :] for i in mu_idx]
    
#initialize Cov Sigma
Sigma = [np.eye(d)]*5

for iter_ in range(10):
    #E-step
    n = X.shape[0]
    
    q_c = []
    for i in range(n):

        q = []
        for kk in range(k):
            q.append(Normal.pdf(X[i,:], mu[kk], Sigma[kk]))
        q = np.array(q)        
        sum_q = np.sum(q*pi)
        q_ci = pi*q/sum_q        
        q_c.append(q_ci)
        
    q_c = np.array(q_c)    
    # M-step
    
    n_k = np.sum(q_c, axis = 0)
    pi = n_k/n
    
    
    for kk in range(k):
        q_k = q_c[:,kk]
        sum_qx = np.sum(X*np.expand_dims(q_k, axis = -1), axis = 0)
        mu_k = 1/n_k[kk]*sum_qx
        mu[kk] = mu_k
        #calculate Sigma_k
        sum_sig = []
        for i in range(n):
        
            sum_sig.append(q_k[i]*np.outer((X[i,:] - mu[kk]), (X[i,:]-mu[kk])))
        
        Sigma[kk] = 1/n_k[kk]*np.sum(np.array(sum_sig), axis = 0)
        
        
    file = "pi-{}.csv".format(iter_+1)
    np.savetxt(file, pi, delimiter=",")
    file = "mu-{}.csv".format(iter_+1)
    np.savetxt(file, mu, delimiter=",")  
    for cluster in range(k):
        file = "Sigma-{}-{}.csv".format(cluster+1, iter_+1)
        np.savetxt(file, Sigma[cluster], delimiter=",")    
    
    
    
    
    
    
        
        
        






























    



