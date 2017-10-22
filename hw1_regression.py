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
    sig_pos = []
    for i in range(X_train.shape[0]): 
        x = X_train[i]
        n, = x.shape
        x.shape = (n, 1)
        sig2 = np.dot(np.dot(x.T, Sigma), x)
        sig_pos.append(sig2)
    
    sig_unsorted = copy.copy(sig_pos)
    sig_pos.sort()
    idx = []
    for each in range(10):
        idx.append(sig_unsorted.index(sig_pos[each]) + 1)
    
    return idx
        
    
        

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file


