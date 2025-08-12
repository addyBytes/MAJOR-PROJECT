import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # <--- CHANGE 1: NEW IMPORT ADDED HERE

# (ATTACK CATEGORY DICTIONARIES REMAIN THE SAME)
ATTACK_CATEGORIES_19 = {
    'ARP_Spoofing': 'Spoofing', 'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood', 'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood', 'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan', 'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan', 'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP', 'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP', 'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP', 'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP', 'TCP_IP-DoS-UDP': 'DoS-UDP', 'Benign': 'Benign'
}
ATTACK_CATEGORIES_6 = {
    'Spoofing': 'Spoofing', 'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT', 'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT', 'MQTT-Malformed_Data': 'MQTT', 'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon', 'Recon-Port_Scan': 'Recon', 'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS', 'DDoS-SYN': 'DDoS', 'DDoS-TCP': 'DDoS', 'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS', 'DoS-SYN': 'DoS', 'DoS-TCP': 'DoS', 'DoS-UDP': 'DoS', 'Benign': 'Benign'
}
ATTACK_CATEGORIES_2 = {
    'ARP_Spoofing': 'attack', 'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack', 'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack', 'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack', 'Recon-Ping_Sweep': 'attack', 'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack', 'TCP_IP-DDoS-ICMP': 'attack', 'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack', 'TCP_IP-DDoS-UDP': 'attack', 'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack', 'TCP_IP-DoS-TCP': 'attack', 'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}


def get_attack_category(file_name, class_config):
    if class_config == 2: categories = ATTACK_CATEGORIES_2
    elif class_config == 6: categories = ATTACK_CATEGORIES_6
    else: categories = ATTACK_CATEGORIES_19
    for key in categories:
        if key in file_name:
            return categories[key]
    return "Unknown"

def load_and_preprocess_data(data_dir, class_config):
    train_files = [f"{data_dir}/train/{f}" for f in os.listdir(f"{data_dir}/train") if f.endswith('.csv')]
    test_files = [f"{data_dir}/test/{f}" for f in os.listdir(f"{data_dir}/test") if f.endswith('.csv')]

    print("[INFO] Loading and sampling 20% of the data...")
    train_df = pd.concat([pd.read_csv(f).assign(file=f) for f in train_files], ignore_index=True).sample(frac=0.2, random_state=42)
    test_df = pd.concat([pd.read_csv(f).assign(file=f) for f in test_files], ignore_index=True).sample(frac=0.2, random_state=42)

    print("[INFO] Creating labels from filenames...")
    train_df['Attack_Type'] = train_df['file'].apply(lambda path: get_attack_category(os.path.basename(path), class_config))
    test_df['Attack_Type'] = test_df['file'].apply(lambda path: get_attack_category(os.path.basename(path), class_config))

    print("[INFO] Splitting data into features and labels...")
    X_train = train_df.drop(['Attack_Type', 'file'], axis=1)
    y_train = train_df['Attack_Type']
    X_test = test_df.drop(['Attack_Type', 'file'], axis=1)
    y_test = test_df['Attack_Type']

    print("[INFO] Cleaning data: replacing infinity and filling NaN values...")
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.median(), inplace=True)
    X_test.fillna(X_test.median(), inplace=True)

    print("[INFO] Performing feature selection...")
    rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    rfe = RFE(estimator=rf, n_features_to_select=25)
    print("[INFO] Fitting the feature selector on training data...")
    rfe.fit(X_train, y_train)
    print("[INFO] Transforming train and test data...")
    X_train = rfe.transform(X_train)
    X_test = rfe.transform(X_test)
    print(f"[INFO] Reduced features to {X_train.shape[1]}")

    print("[INFO] Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # --- CHANGE 2: BALANCE THE DATASET WITH SMOTE ---
    print("[INFO] Balancing the training data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_encoded)
    X_train = X_train_balanced
    y_train_encoded = y_train_balanced
    print(f"[INFO] Data balanced. New training samples: {X_train.shape[0]}")
    # --- END OF CHANGE ---

    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)

    print("[INFO] Creating validation split...")
    X_train, X_val, y_train_categorical, y_val_categorical = train_test_split(
        X_train, y_train_categorical, test_size=0.2, random_state=42
    )

    print("[INFO] Scaling data...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("[INFO] Reshaping data for the model...")
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder, scaler