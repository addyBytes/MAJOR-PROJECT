import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os

# Define attack categories
ATTACK_CATEGORIES_19 = { 
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_6 = {  
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_2 = {  
    'ARP_Spoofing': 'attack',
    'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack',
    'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack',
    'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack',
    'Recon-Ping_Sweep': 'attack',
    'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack',
    'TCP_IP-DDoS-ICMP': 'attack',
    'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack',
    'TCP_IP-DDoS-UDP': 'attack',
    'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack',
    'TCP_IP-DoS-TCP': 'attack',
    'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}

def get_attack_category(file_name, class_config): 
    """Get attack category from file name."""
    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories = ATTACK_CATEGORIES_6
    else:  # Default to 19 classes 
        categories = ATTACK_CATEGORIES_19  
    
    for key in categories:
        if key in file_name:
            return categories[key]
    
    # If no match found, assume benign
    return 'Benign' if class_config != 2 else 'Benign'

def load_data_in_chunks(files, class_config, sample_size=100000):
    """Load data in chunks to handle memory efficiently."""
    all_chunks = []
    
    for file_path in files:
        print(f"Processing: {os.path.basename(file_path)}")
        try:
            # Read file info first
            df_info = pd.read_csv(file_path, nrows=1)
            total_rows = sum(1 for line in open(file_path)) - 1  # -1 for header
            
            # If file is small, load all
            if total_rows <= sample_size:
                df = pd.read_csv(file_path)
                df['Attack_Type'] = get_attack_category(file_path, class_config)
                all_chunks.append(df)
                print(f"  Loaded all {len(df)} rows")
            else:
                # Sample from large files
                skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                                  total_rows - sample_size, 
                                                  replace=False))
                df = pd.read_csv(file_path, skiprows=skip_rows)
                df['Attack_Type'] = get_attack_category(file_path, class_config)
                all_chunks.append(df)
                print(f"  Sampled {len(df)} rows from {total_rows} total rows")
                
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            continue
    
    if not all_chunks:
        raise ValueError("No data loaded successfully!")
    
    # Combine chunks efficiently
    print("Combining data chunks...")
    combined_df = pd.concat(all_chunks, ignore_index=True)
    
    # Clean up memory
    del all_chunks
    
    return combined_df

def load_and_preprocess_data(data_dir, class_config, k_features=50, max_samples_per_class=50000):
    """Load, preprocess, and prepare data for training with memory optimization."""
    print("Loading data with memory optimization...")
    
    # Get file lists
    train_files = [f"{data_dir}/train/{f}" for f in os.listdir(f"{data_dir}/train") if f.endswith('.csv')]
    test_files = [f"{data_dir}/test/{f}" for f in os.listdir(f"{data_dir}/test") if f.endswith('.csv')]
    
    print(f"Found {len(train_files)} training files and {len(test_files)} test files")
    
    # Load data in chunks
    train_df = load_data_in_chunks(train_files, class_config, sample_size=100000)
    test_df = load_data_in_chunks(test_files, class_config, sample_size=50000)
    
    print(f"Initial training data shape: {train_df.shape}")
    print(f"Initial test data shape: {test_df.shape}")
    
    # Remove rows with missing Attack_Type
    train_df = train_df.dropna(subset=['Attack_Type']).copy()
    test_df = test_df.dropna(subset=['Attack_Type']).copy()
    
    print(f"After removing missing labels - Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Class distribution in training:\n{train_df['Attack_Type'].value_counts()}")
    
    # Balance classes by sampling
    print("Balancing dataset...")
    balanced_dfs = []
    for class_name in train_df['Attack_Type'].unique():
        class_df = train_df[train_df['Attack_Type'] == class_name]
        if len(class_df) > max_samples_per_class:
            class_df = class_df.sample(n=max_samples_per_class, random_state=42)
        balanced_dfs.append(class_df)
    
    train_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Similarly balance test set (smaller sample)
    balanced_test_dfs = []
    for class_name in test_df['Attack_Type'].unique():
        class_df = test_df[test_df['Attack_Type'] == class_name]
        if len(class_df) > max_samples_per_class // 4:  # Smaller test set
            class_df = class_df.sample(n=max_samples_per_class // 4, random_state=42)
        balanced_test_dfs.append(class_df)
    
    test_df = pd.concat(balanced_test_dfs, ignore_index=True)
    
    print(f"After balancing - Train: {train_df.shape}, Test: {test_df.shape}")
    print(f"Balanced class distribution:\n{train_df['Attack_Type'].value_counts()}")
    
    # Prepare features and labels
    X_train = train_df.drop(['Attack_Type'], axis=1)
    y_train = train_df['Attack_Type']
    X_test = test_df.drop(['Attack_Type'], axis=1)
    y_test = test_df['Attack_Type']
    
    # Keep only numeric columns
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[numeric_columns]
    X_test = X_test[numeric_columns]
    
    # CRITICAL FIX: Clean the data properly
    print("Cleaning data (removing inf/nan/extreme values)...")
    
    # Replace infinite values with NaN first
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Fill missing values with column median (more robust than 0)
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test too
    
    # Clip extreme values (anything beyond 99th percentile)
    for col in X_train.columns:
        q99 = X_train[col].quantile(0.99)
        q01 = X_train[col].quantile(0.01)
        X_train[col] = X_train[col].clip(lower=q01, upper=q99)
        X_test[col] = X_test[col].clip(lower=q01, upper=q99)
    
    # Convert to float32 to save memory
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    
    # Final check for any remaining problematic values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Ensure no infinite values remain
    X_train = np.where(np.isfinite(X_train), X_train, 0)
    X_test = np.where(np.isfinite(X_test), X_test, 0)
    
    print(f"Feature shape after preprocessing: {X_train.shape}")
    print(f"Data range - Min: {X_train.min():.4f}, Max: {X_train.max():.4f}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"Classes: {label_encoder.classes_}")
    
    # Feature Selection using SelectKBest
    print(f"Applying feature selection (top {k_features} features)...")
    k_features = min(k_features, X_train.shape[1])
    
    # Use a safer feature selector that handles constant features
    selector = SelectKBest(score_func=f_classif, k=k_features)
    
    try:
        X_train_selected = selector.fit_transform(X_train, y_train_encoded)
        X_test_selected = selector.transform(X_test)
    except Exception as e:
        print(f"Feature selection warning: {e}")
        print("Using all features instead...")
        X_train_selected = X_train
        X_test_selected = X_test
        selector = None
    
    print(f"Selected features shape: {X_train_selected.shape}")
    
    # Split training data for validation
    X_train_sel, X_val_sel, y_train_enc, y_val_enc = train_test_split(
        X_train_selected, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
    )
    
    # Handle class balancing - try SMOTE first, fallback to class weights only
    print("Attempting SMOTE for class balancing...")
    
    try:
        # Get current distribution
        unique, counts = np.unique(y_train_enc, return_counts=True)
        print(f"Original distribution: {dict(zip(unique, counts))}")
        
        # Use simple minority oversampling for binary classification
        if class_config == 2:
            smote = SMOTE(random_state=42, sampling_strategy='minority', k_neighbors=min(3, min(counts)-1))
        else:
            smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=3)
        
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_sel, y_train_enc)
        print("SMOTE applied successfully!")
        
    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Skipping SMOTE - using original data with class weights only...")
        X_train_balanced, y_train_balanced = X_train_sel, y_train_enc
    
    print(f"Final training shape: {X_train_balanced.shape}")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print(f"Final class distribution: {dict(zip(unique, counts))}")
    
    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced).astype('float32')
    X_val_scaled = scaler.transform(X_val_sel).astype('float32')
    X_test_scaled = scaler.transform(X_test_selected).astype('float32')
    
    # Convert to categorical
    y_train_categorical = to_categorical(y_train_balanced)
    y_val_categorical = to_categorical(y_val_enc)
    y_test_categorical = to_categorical(y_test_encoded)
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)
    X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
    
    print("Data preprocessing completed successfully!")
    print(f"Training data shape: {X_train_reshaped.shape}")
    print(f"Validation data shape: {X_val_reshaped.shape}")
    print(f"Test data shape: {X_test_reshaped.shape}")
    
    return (X_train_reshaped, X_val_reshaped, X_test_reshaped, 
            y_train_categorical, y_val_categorical, y_test_categorical, 
            label_encoder, selector, scaler)
