import numpy as np
import sys
import copy

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1(lambda_ = lambda_input):
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    
    I = np.diag([1]*X_train.shape[1])  
    wRR = np.dot(np.dot(np.linalg.inv(lambda_*I + np.dot(X_train.T, X_train)), X_train.T), y_train)
    
    return wRR

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2(lambda_ = lambda_input, sigma2 = sigma2_input):
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    I = np.diag([1]*X_train.shape[1])
    Sigma = np.linalg.inv(lambda_*I + 1/sigma2*np.dot(X_train.T, X_train))
    idx = []
    picked = set()
    x_dict = {}
    for i in range(X_test.shape[0]):
        x = copy.copy(tuple(X_test[i,:]))
        x_dict[x] = i+1
        
    for i in range(10): 

        sig_dict = dict()
        for ii in range(X_test.shape[0]):
            x = X_test[ii]
            x_ = tuple(x)
            if x_ not in picked:
                
                n, = x.shape
                x.shape = (n, 1)
                sig2 = np.dot(np.dot(x.T, Sigma), x)[0]
                sig_dict[float(sig2)] = x_
        s_list = list(sig_dict.keys())  
        s_list.sort()
        max_ = s_list[-1]
        x0 = sig_dict[max_]
        picked.add(x0)
        
        idx.append(x_dict[x0])
        Sigma = np.linalg.inv(np.linalg.inv(Sigma) + (1/sigma2)*np.outer(np.array(x0),np.array(x0)))
            
  
    return idx
        
    
        

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file


