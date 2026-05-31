import numpy as np
import copy

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))

    cache = (z)

    return s, cache


def relu(z):
    r = np.maximum(0, z)
    cache = (z)
    return r, cache

def sigmoid_backward(dA, cache):
    z = cache
    s = s = 1 / (1 + np.exp(-z))
    dZ = dA * (s*(1-s))
    return dZ
def relu_backward(dA, cache):
    z = cache
    dZ = np.array(dA, copy=True)
    # When z <= 0, the derivative of ReLU is 0, so set those gradients to 0
    dZ[z <= 0] = 0
    return dZ

#initialization (manual way of writing a three layer neural network)
# def initialize_parameters(nx, k1, k2, ny):
#     """
#     nx: number of nodes in the 0th layer/ number of features in X.
#     k1: number of nodes in the 1st hidden layer.
#     k2: number of nodes in the 2nd hidden layer.
#     ny: size of the output layer.

#     Returns: params
#         W1: weight for the 1st hidden layer (k1, nx).
#         b1: bias for the 1st hidden layer (k1, 1).
#         W2: weight for the 2nd hidden layer (k2, k1).
#         b2: bias for the 2nd hidden layer (k2, 1).
#         W3: weight for the 3rd hidden layer (ny, k2).
#         b3: bias for the 3rd hidden layer (ny, 1).
        
#     """
#     # to stop randomising the weights every time we rerun.
#     np.random.seed(42)

#     #initializing weights and bias 
#     w1 = np.random.randn(k1, nx)*0.01
#     b1 = np.random.zeros((k1, 1))
#     w2 = np.random.randn(k2, k1)*0.01
#     b2 = np.random.zeros((k2, 1))
#     w3 = np.random.randn(ny, k2)*0.01
#     b3 = np.random.zeros((ny, 1))

#     params = {
#         'w1' : w1,
#         'b1': b1,
#         'w2' : w2,
#         'b2': b2,
#         'w3' : w3,
#         'b3': b3
#     }
#     return params

#initialization of an L-layered neural network
def initialize_parameters(layer_dims):
    """
    layer_dims: a python array containg dimesions of each layer.
    eg: layer_dims = [nx, k1, k2, ... ,ny]                                    
                              
    Returns: params
        W1: weight for the 1st hidden layer (k1, nx).
        b1: bias for the 1st hidden layer (k1, 1).
        W2: weight for the 2nd hidden layer (k2, k1).
        b2: bias for the 2nd hidden layer (k2, 1).
        W3: weight for the 3rd hidden layer (ny, k2).
        b3: bias for the 3rd hidden layer (ny, 1).
    """
    # to stop randomising the weights every time we rerun.
    np.random.seed(42)

    params = {}
    L = len(layer_dims)

    for l in range(1, L): # start from 1 cause we have number of features in the 0th pos.
        params['w' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01 #(k1, nx) for 1st iteration
        params['b' + str(l)] = np.zeros((layer_dims[l], 1)) # (k1 , 1) for 1st iteration

        # to be safe and not mismatch the shapes
        assert(params['w' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(params['b' + str(l)].shape == (layer_dims[l], 1))
    return params

def forward(A, w, b):
    """
    returns Z
    A: (size of previous layer, number of training examples)
    """
    Z = np.dot(w, A) + b

    cache = (A, w, b) # required when we backpropogate

    return Z, cache

def activation_forward(A_prev, w, b, act):
    """
    apply activation after every forward step
    """
    activation = {
        'sigmoid': sigmoid,
        'relu': relu
    }

    if act in activation:
        Z, linear_cache = forward(A_prev, w, b)
        A, activation_cache = activation[act](Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def repeat_activation_forward(X, params):
    caches = []
    L = len(params) // 2  # Total number of layers (6)
    A = X

    # 1. Hidden Layers 1 through L-1: Safe ReLU tracking
    for l in range(1, L):
        A_prev = A
        A, cache = activation_forward(A_prev, params['w' + str(l)], params['b' + str(l)], 'relu')
        caches.append(cache)

    # 2. Final Output Embedding Layer L: Direct linear project (No ReLU)
    A_prev = A
    Z, linear_cache = forward(A_prev, params['w' + str(L)], params['b' + str(L)])

    # CRITICAL FIX: Save a dummy string for the activation cache
    # to prevent relu_backward from accidentally reading Z later!
    cache = (linear_cache, "LINEAR_LAYER_NO_RELU")
    A = Z
    caches.append(cache)

    return A, caches
        
        
def compute_cost(AL, Y):
    """
    Y => (ny, m)
    """

    m = Y.shape[1]

    cost = -(1/m) * (np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)))

    cost = np.squeeze(cost)

    return cost

def backward(dZ, cache): # cache: (A, w, b) from the forward function
    A_prev, w, b = cache

    m = A_prev.shape[1]

    dw = (1/m)*(np.dot(dZ, A_prev.T))

    db = (1/m)*(np.sum(dZ, axis = 1, keepdims = True))

    dA_prev = np.dot(w.T, dZ)

    return dA_prev, dw, db
    
def activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dw, db = backward(dZ, linear_cache)
    else:
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dw, db = backward(dZ, linear_cache)

    return dA_prev, dw, db

def repeat_activation_backward(dAL, caches):
    grads = {}
    L = len(caches)
    dA_current = dAL

    # 1. Final Output Layer (Layer L): Direct linear backpropagation
    current_cache = caches[L - 1]
    linear_cache, activation_check = current_cache

    # Directly pass back through the linear layer equations
    dZ = dA_current
    dA_prev_temp, dw_temp, db_temp = backward(dZ, linear_cache)

    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dw" + str(L)] = dw_temp
    grads["db" + str(L)] = db_temp

    # Update current gradient to flow down to hidden layers
    dA_current = dA_prev_temp

    # 2. Hidden Layers (L-1 down to 1): Safe ReLU backpropagation
    for l in reversed(range(L - 1)):
        current_cache = caches[l]

        # Pull the tracking metrics cleanly
        dA_prev_temp, dw_temp, db_temp = activation_backward(dA_current, current_cache, 'relu')

        grads["dA" + str(l)] = dA_prev_temp
        grads["dw" + str(l + 1)] = dw_temp
        grads["db" + str(l + 1)] = db_temp

        dA_current = dA_prev_temp

    return grads


    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dw_temp, db_temp = activation_backward(dA_current, current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dw" + str(l+1)] = dw_temp
        grads["db" + str(l+1)] = db_temp

    return grads

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters)//2
    for l in range(1, L+1):
        parameters['w' + str(l)] -= learning_rate*grads['dw' + str(l)]
        parameters['b' +str(l)] -= learning_rate*grads['db' + str(l)]
    return parameters