import numpy as np
import pandas as pd
import pickle
import model as nn  

def run_blind_validation_test():
    print("🧪 Starting Blind Validation Test...")

    X_user_test = pd.read_csv('Test/scaled_user_features_test.csv', index_col=0).to_numpy().T
    X_house_test = pd.read_csv('Test/scaled_house_features_test.csv', index_col=0).to_numpy().T


    Y_df = pd.read_csv('Test/market_interactions_test.csv', index_col=0)

    Y_test_true = Y_df.iloc[:, -1:].to_numpy().T

    print(f"📋 Loaded Test Shapes - Users: {X_user_test.shape} | Houses: {X_house_test.shape} | Labels: {Y_test_true.shape}")


    print("📂 Loading trained model parameters...")
    with open('regularised_trained_user_params.pkl', 'rb') as f:
        user_params = pickle.load(f)
    with open('regularised_trained_house_params.pkl', 'rb') as f:
        house_params = pickle.load(f)


    A1_test, _ = nn.repeat_activation_forward(X_user_test, user_params, keep_prob = 1.0)
    A2_test, _ = nn.repeat_activation_forward(X_house_test, house_params, keep_prob = 1.0)


    Z_test = np.dot(A1_test.T, A2_test)
    AL_test, _ = nn.sigmoid(Z_test)



    AL_test_flattened = AL_test.ravel().reshape(1, -1)


    test_predictions = (AL_test_flattened > 0.35).astype(int)


    test_accuracy = np.mean(test_predictions == Y_test_true) * 100
    test_cost = nn.compute_cost(AL_test_flattened, Y_test_true, params={**user_params, **house_params}, lambd=0.1)

    print("\n================ TEST RESULTS ================")
    print(f"📉 Fresh Test Set Cost:     {test_cost:.6f}")
    print(f"🎯 Fresh Test Set Accuracy: {test_accuracy:.2f}%")
    print("==============================================")

    return test_accuracy

if __name__ == "__main__":
    run_blind_validation_test()