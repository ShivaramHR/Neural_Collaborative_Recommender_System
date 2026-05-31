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
    user_layer_dims = [X_user.shape[0], 128, 128, 64, 64, 32, 1]
    house_layer_dims = [X_house.shape[0], 128, 128, 64, 64, 32, 1]

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
        AL, _ = nn.sigmoid(Z)

        #Evaluate the cost
        cost = nn.compute_cost(AL, Y)

        #Backward Pass
        #directly used the result derived.
        dZ = AL - Y

        # calculate dA1 and dA2 using chain rule
        dA1 = dZ * A2
        dA2 = dZ * A1

        # use the repeat_activation_backward function written in model.py
        user_grads = nn.repeat_activation_backward(dA1, caches1)
        house_grads = nn.repeat_activation_backward(dA2, caches2)

        #update parameters
        user_params = nn.update_parameters(user_params, user_grads, learning_rate)
        house_params = nn.update_parameters(house_params, house_grads, learning_rate)

        # Print status updates
        if epoch == 1 or epoch % 100 == 0:
            # Quick metric calculation: Round probabilities to get predictions
            predictions = (AL > 0.5).astype(int)
            accuracy = np.mean(predictions == Y) * 100
            print(f"Epoch {epoch:4d}/{epochs} | Cost: {cost:.6f} | Training Accuracy: {accuracy:.2f}%")

    return user_params, house_params

if __name__ == "__main__":
    #Load the data
    X_user, X_house, Y = load_and_prepare_data()

    #train the model
    trained_user_model, trained_house_model = train(X_user, X_house, Y, 0.001, 1000)

    print("Model optimized successfully!")