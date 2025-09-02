import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
import numpy as np
import os

def focal_loss(gamma=2., alpha=0.75):
    """Focal loss for imbalanced classification."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = -alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed

def create_cnn_lstm_model(input_shape, num_classes):
    """Create and compile the CNN-LSTM hybrid model with focal loss."""
    model = Sequential()
    
    # CNN layers for feature extraction
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # LSTM layers for sequence learning
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.4))
    
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.4))
    
    # Dense layers for classification
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile with focal loss for better imbalanced classification
    optimizer = Adam(learning_rate=0.001)
    
    if num_classes == 2:
        # Use focal loss for binary classification
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(gamma=2.0, alpha=0.75),  # Alpha favors minority class
            metrics=['accuracy', 'precision', 'recall']
        )
        print("Using Focal Loss for imbalanced binary classification")
    else:
        # Use standard loss for multi-class
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        print("Using standard categorical crossentropy")
    
    return model

def calculate_class_weights(y_train_categorical, class_config):
    """Calculate class weights with HEAVY emphasis on Benign class."""
    
    y_train_labels = np.argmax(y_train_categorical, axis=1)
    classes = np.unique(y_train_labels)
    
    # For binary classification, give MUCH more weight to Benign class
    if class_config == 2 and len(classes) == 2:
        # HEAVY weights - 5x for Benign detection
        class_weights = {0: 5.0, 1: 1.0}  # 5x weight for Benign class
        print(f"Using HEAVY class weights for Benign detection: {class_weights}")
    else:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_labels)
        class_weights = dict(zip(classes, class_weights))
        print(f"Using computed balanced weights: {class_weights}")
    
    return class_weights

def train_model(model, X_train, y_train_categorical, X_val, y_val_categorical, 
                epochs=50, batch_size=64, class_config=2, save_path="models/"):
    """Train the CNN-LSTM model with callbacks and save the best model."""
    
    # Create models directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train_categorical, class_config)
    
    # Define model save path
    model_save_path = os.path.join(save_path, f"best_cnn_lstm_model_{class_config}class.h5")
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=6,
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpoint to save the best model
    model_checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr, model_checkpoint]
    
    print(f"Training model... Will save best model to: {model_save_path}")
    
    # Train the model
    history = model.fit(
        X_train, y_train_categorical,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_categorical),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Training completed! Best model saved to: {model_save_path}")
    
    return model, history
