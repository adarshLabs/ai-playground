import numpy as np
def relu(input):
    return np.maximum(0, input)

def sigmoid(input):
    #return 1/(1 + np.exp(-input))
    return 1 / (1 + np.exp(-np.clip(input, -500, 500)))   
           
def tanh(input):
    return (np.exp(input) - np.exp(-input))/(np.exp(input) + np.exp(-input))
