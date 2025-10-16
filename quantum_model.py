# models/quantum_model.py (FINAL VERSION with Advanced Ansatz)

import pandas as pd
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import TwoLocal # <- NEW: Import for the advanced circuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.optimize import minimize

# --- MODEL 1: QUANTUM REGRESSOR ---
# This model remains unchanged as we are focusing the experiment on the classifier.
def train_and_predict_quantum_regressor(data: pd.DataFrame, n_future=30):
    target = data['Close'].diff().dropna().values
    features = data.iloc[1:].drop(columns=['Close', 'Stock Splits', 'Dividends'], errors='ignore').values
    
    feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = feature_scaler.fit_transform(features)
    y_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]
    
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    ansatz_params = [Parameter(f'θ_{i}') for i in range(num_features)]
    ansatz = QuantumCircuit(num_features)
    for i in range(num_features):
        ansatz.ry(ansatz_params[i], i)
    circuit = feature_map.compose(ansatz)
    observable = SparsePauliOp("Z" * num_features)
    estimator = Estimator()

    def cost_function_mse(params, X, y):
        param_sets = [np.concatenate((x_i, params)) for x_i in X]
        job = estimator.run([circuit] * len(X), [observable] * len(X), param_sets)
        predictions_scaled = job.result().values
        return mean_squared_error(y, predictions_scaled)

    initial_params = np.random.uniform(0, 2*np.pi, size=num_features)
    res = minimize(cost_function_mse, initial_params, args=(X_train, y_train), method='COBYLA', options={'maxiter': 50})
    trained_params = res.x

    future_data_points = X_test[-n_future:]
    future_param_sets = [np.concatenate((x_f, trained_params)) for x_f in future_data_points]
    job = estimator.run([circuit] * len(future_data_points), [observable] * len(future_data_points), future_param_sets)
    future_predictions_scaled = job.result().values
    
    predictions_actual_change = target_scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    test_param_sets = [np.concatenate((x_t, trained_params)) for x_t in X_test]
    job = estimator.run([circuit] * len(X_test), [observable] * len(X_test), test_param_sets)
    test_predictions_scaled = job.result().values
    mse = mean_squared_error(y_test, test_predictions_scaled)
    
    return mse, predictions_actual_change.flatten().tolist()


# --- MODEL 2: QUANTUM CLASSIFIER (This function is now updated) ---

def train_and_predict_quantum_classifier(data: pd.DataFrame, n_future=30):
    features = data.drop(columns=['Close', 'Stock Splits', 'Dividends'], errors='ignore').values
    target_binary = np.where(data['Close'].diff() > 0, 1, 0)[1:]
    features = features[1:]
    
    feature_scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = feature_scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, target_binary, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]
    
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1)
    
    # --- CHANGE 1: REPLACE THE OLD, SIMPLE ANSATZ ---
    # The old, simple ansatz is replaced with the more powerful TwoLocal circuit.
    # This circuit adds entanglement (with 'cz' gates) between the qubits,
    # which can help the model learn more complex patterns.
    ansatz = TwoLocal(num_features, 'ry', 'cz', reps=2, insert_barriers=True)
    
    circuit = feature_map.compose(ansatz)
    observable = SparsePauliOp("Z" * num_features)
    estimator = Estimator()

    def cost_function_accuracy(params, X, y):
        param_sets = [np.concatenate((x_i, params)) for x_i in X]
        job = estimator.run([circuit] * len(X), [observable] * len(X), param_sets)
        predictions = job.result().values
        pred_binary = np.where(np.array(predictions) > 0, 1, 0)
        return 1 - accuracy_score(y, pred_binary)

    # --- CHANGE 2: GET NUMBER OF PARAMETERS AUTOMATICALLY ---
    # The number of trainable weights is now determined by the ansatz itself.
    num_params = ansatz.num_parameters
    initial_params = np.random.uniform(-np.pi, np.pi, size=num_params)
    
    print(f"Starting quantum classifier training with advanced TwoLocal circuit ({num_params} parameters)...")
    res = minimize(cost_function_accuracy, initial_params, args=(X_train, y_train), method='COBYLA', options={'maxiter': 50})
    trained_params = res.x
    
    # --- The rest of the function operates the same way ---
    future_data_points = X_test[-n_future:]
    future_param_sets = [np.concatenate((x_f, trained_params)) for x_f in future_data_points]
    job = estimator.run([circuit] * len(future_data_points), [observable] * len(future_data_points), future_param_sets)
    quantum_predictions_raw = job.result().values
    quantum_predictions = [1 if val > 0 else 0 for val in quantum_predictions_raw]
    
    test_param_sets = [np.concatenate((x_t, trained_params)) for x_t in X_test]
    job = estimator.run([circuit] * len(X_test), [observable] * len(X_test), test_param_sets)
    test_predictions_raw = job.result().values
    test_predictions = [1 if val > 0 else 0 for val in test_predictions_raw]
    accuracy = accuracy_score(y_test, test_predictions)
    
    print("Quantum classifier training complete.")
    return accuracy, quantum_predictions