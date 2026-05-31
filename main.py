import numpy as np
import pandas as pd
import model as nn

def load_and_prepare_data():

    print('Loading and preparing data...')
    X_user_df = pd.read_csv('scaled_user_features.csv')
    X_house_df = pd.read_csv('scaled_house_features.csv')
    Y_df = pd.read_csv('market_interactions.csv')

    #convert to numpy arrays
    X_user = X_user_df.to_numpy().T
    X_house = X_house_df.to_numpy().T
    Y = Y_df.to_numpy().T

    print(f"✅ Data Loaded. User Shapes: {X_user.shape}, House Shapes: {X_house.shape}, Labels Shape: {Y.shape}")

    return X_user, X_house, Y
# X_user, X_house, Y = load_and_prepare_data()


def train(X_user, X_house, Y, learning_rate, epochs):
    # assign the structure of the neural network 6 layers of relu and 1 layer of sigmoid at the end.
    user_layer_dims = [X_user.shape[0], 128, 64, 64, 32, 32, 32]
    house_layer_dims = [X_house.shape[0], 128, 64, 64, 32, 32, 32]

    #initialize parameters
    user_params = nn.initialize_parameters(user_layer_dims)
    house_params = nn.initialize_parameters(house_layer_dims)

    # number of iterations for training
    for epoch in range(1, epochs + 1):
        #forward pass
        A1, caches1 = nn.repeat_activation_forward(X_user, user_params)
        A2, caches2 = nn.repeat_activation_forward(X_house, house_params)

        # calculate the dot product of A1 and A2
        Z = np.sum(A1*A2, axis=0, keepdims=True)
        AL, cache = nn.sigmoid(Z)

        #Evaluate the cost
        cost = nn.compute_cost(AL, Y)

        #Backward Pass



    return 

