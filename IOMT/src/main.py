import os
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib

# Import your custom functions
from data_loader import load_and_preprocess_data
from model import create_model

# --- 1. CONFIGURATION ---
print("[INFO] Setting up project configuration...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_DIR = r"c:\Users\Aditya\Downloads\IoMT_Threat_Detection\Data"
CLASS_CONFIG = 2
TUNER_DIR = 'tuner_results'

# --- 2. LOAD AND PREPARE THE DATA (WITH SMOTE FIX) ---
print("[INFO] Loading and preprocessing data...")
(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    label_encoder, scaler
) = load_and_preprocess_data(DATA_DIR, CLASS_CONFIG)

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y_train.shape[1]

# --- 3. LOAD BEST HYPERPARAMETERS FROM PREVIOUS RUN ---
print("\n[INFO] Loading best hyperparameters from the previous tuner search...")
model_builder = lambda hp: create_model(hp, input_shape=input_shape, num_classes=num_classes)
tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=5,
    directory=TUNER_DIR,
    project_name='iomt_threat_detection'
)

# This line loads the best settings without re-running the long search
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
[INFO] Loaded best settings from previous run.
[INFO] Best CNN filters: {best_hps.get('cnn_filters')}
[INFO] Best LSTM units: {best_hps.get('lstm_units')}
[INFO] Best learning rate: {best_hps.get('learning_rate')}
""")

# --- 4. TRAIN THE FINAL MODEL ON BALANCED DATA ---
print("\n[INFO] Training the final, optimized model on the newly balanced data...")
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val)
)

# --- 5. EVALUATE THE FINAL MODEL ---
print("\n[INFO] Evaluating the final model on the test set...")
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

print("\n[INFO] Generating classification report and confusion matrix...")
y_pred_probs = best_model.predict(X_test)
import os
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib

# Import your custom functions
from data_loader import load_and_preprocess_data
from model import create_model

# --- 1. CONFIGURATION ---
print("[INFO] Setting up project configuration...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DATA_DIR = r"c:\Users\Aditya\Downloads\IoMT_Threat_Detection\Data"
CLASS_CONFIG = 2
TUNER_DIR = 'tuner_results'

# --- 2. LOAD AND PREPARE THE DATA (WITH FINAL FIX) ---
print("[INFO] Loading and preprocessing data...")
(
    X_train, X_val, X_test, 
    y_train, y_val, y_test, 
    label_encoder, scaler
) = load_and_preprocess_data(DATA_DIR, CLASS_CONFIG)

input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = y_train.shape[1]

# --- 3. RUN THE FULL TUNER SEARCH (NO LONGER FAST-TRACK) ---
# We need to run a fresh search because the balanced data is different
print("\n[INFO] Starting a fresh hyperparameter search for the balanced data...")
model_builder = lambda hp: create_model(hp, input_shape=input_shape, num_classes=num_classes)
tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory=TUNER_DIR,
    project_name='iomt_balanced_detection', # New project name to avoid conflicts
    overwrite=True # Start a fresh search
)

# Start the search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
[INFO] Hyperparameter search complete.
[INFO] Best CNN filters: {best_hps.get('cnn_filters')}
[INFO] Best LSTM units: {best_hps.get('lstm_units')}
[INFO] Best learning rate: {best_hps.get('learning_rate')}
""")

# --- 4. TRAIN THE BEST MODEL ---
print("\n[INFO] Training the best model with optimal hyperparameters...")
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_val, y_val)
)

# --- 5. EVALUATE THE FINAL MODEL ---
print("\n[INFO] Evaluating the final model on the test set...")
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

print("\n[INFO] Generating classification report and confusion matrix...")
y_pred_probs = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
target_names = label_encoder.classes_

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# --- 6. SAVE EVERYTHING ---
print("\n[INFO] Saving the final trained model...")
best_model.save("final_iomt_model.h5")
print("[SUCCESS] Model saved as 'final_iomt_model.h5'")

print("[INFO] Saving the scaler and label encoder...")
joblib.dump(scaler, "scaler.gz")
joblib.dump(label_encoder, "label_encoder.gz")
print("[SUCCESS] Scaler and encoder saved.")
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
target_names = label_encoder.classes_

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))

print("Confusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# --- 6. SAVE EVERYTHING ---
print("\n[INFO] Saving the final trained model...")
best_model.save("final_iomt_model.h5")
print("[SUCCESS] Model saved as 'final_iomt_model.h5'")

print("[INFO] Saving the scaler and label encoder...")
joblib.dump(scaler, "scaler.gz")
joblib.dump(label_encoder, "label_encoder.gz")
print("[SUCCESS] Scaler and encoder saved.")