from __future__ import division
import numpy as np
import sys
from scipy.sparse import coo_matrix
train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

# Implement function here
def PMF(train_data):
    #preprocess the data

    indx = [int(x) for x in train_data[:,0]]
    indy = [int(y) for y in train_data[:,1]]
    m = coo_matrix((train_data[:,2], (indx, indy)))
    m = m.tocsc()

    shape_ = m.shape 
        
        
    u = np.ones([shape_[0], d])
    v = np.ones([shape_[1], d])

    L = []
    U_matrices = []
    V_matrices = []
    for iter_ in range(50):
        for i in range(shape_[0]):
            fac1 = lam*sigma2*np.eye(5) + np.dot(v.T, v)
            fac1 = np.linalg.inv(fac1)
            m_i = m.getrow(i)
            fac2 = m_i.multiply(v.T).sum(axis = 1)
                         
            u[i] = np.dot(fac1,fac2).flatten()
            
        for j in range(shape_[1]):
            fac1 = lam*sigma2*np.eye(5) + np.dot(u.T, u)
            fac1 = np.linalg.inv(fac1)
            m_j = m.getcol(j)
            fac2 = m_j.multiply(u).sum(axis = 0)
            v[j] = np.dot(fac1,fac2.T).flatten()
            
   
        A = np.power((m - np.dot(u, v.T)), 2).sum()
       
        B = 0
        C = 0
        for i in range(shape_[0]):
            B += lam/2*np.dot(u[i], u[i])
        for j in range(shape_[1]):
            C += lam/2*np.dot(v[j], v[j])
            
        L.append(-A-B-C)
        U_matrices.append(u)
        V_matrices.append(v)
    return L, U_matrices, V_matrices





# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")
