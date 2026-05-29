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


