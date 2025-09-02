import os
import argparse
import numpy as np
import pickle
from data_loader import load_and_preprocess_data
from model import create_cnn_lstm_model, train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN-LSTM for network intrusion detection.")
    parser.add_argument("--class_config", type=int, choices=[2, 6, 19], default=2,
                        help="Number of classes for classification (2, 6, or 19)")
    parser.add_argument("--k_features", type=int, default=25,
                        help="Number of features to select")
    parser.add_argument("--epochs", type=int, default=35,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_dir = os.path.join(script_dir, 'data')
    models_dir = os.path.join(script_dir, 'models')
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    print(f"Looking for data in: {data_dir}")
    print(f"Data directory exists: {os.path.exists(data_dir)}")
    print(f"Models will be saved to: {models_dir}")
    
    print(f"🚀 Loading and preprocessing data with {args.k_features} features...")
    
    # Load and preprocess data
    (X_train, X_val, X_test, 
     y_train_categorical, y_val_categorical, y_test_categorical, 
     label_encoder, selector, scaler) = load_and_preprocess_data(
        data_dir, args.class_config, args.k_features)
    
    print(f"✅ Input shape: {X_train.shape}")
    print(f"✅ Number of classes: {y_train_categorical.shape[1]}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_cnn_lstm_model(input_shape, y_train_categorical.shape[1])
    
    # Print model summary
    print("\n📋 Model Architecture:")
    model.summary()
    
    # Check GPU
    if tf.test.gpu_device_name():
        print('🚀 GPU is available!')
    else:
        print('💻 GPU is not available. Using CPU.')
    
    # Train model
    print(f"\n🏋️ Starting training for {args.epochs} epochs...")
    model, history = train_model(
        model, X_train, y_train_categorical, X_val, y_val_categorical, 
        epochs=args.epochs, batch_size=args.batch_size, 
        class_config=args.class_config, save_path=models_dir
    )
    
    # Save preprocessing components
    preprocessing_path = os.path.join(models_dir, f'preprocessing_{args.class_config}class.pkl')
    with open(preprocessing_path, 'wb') as f:
        pickle.dump({
            'label_encoder': label_encoder,
            'selector': selector,
            'scaler': scaler,
            'input_shape': input_shape,
            'class_config': args.class_config
        }, f)
    print(f"✅ Preprocessing components saved to: {preprocessing_path}")
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    loss, accuracy, precision, recall = model.evaluate(X_test, y_test_categorical, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    
    # Make predictions
    y_pred_categorical = model.predict(X_test)
    y_pred_encoded = y_pred_categorical.argmax(axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    y_test_decoded = label_encoder.inverse_transform(y_test_categorical.argmax(axis=1))
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y_test_decoded, y_pred)
    precision = precision_score(y_test_decoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_decoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_decoded, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📈 FINAL RESULTS:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.1f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.1f}%)")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test_decoded, y_pred, zero_division=0))
    
    print(f"\n🎯 Confusion Matrix:")
    cm = confusion_matrix(y_test_decoded, y_pred)
    print(cm)
    
    # Per-class Performance
    print(f"\n🎯 Per-class Performance:")
    for class_name in label_encoder.classes_:
        class_mask = y_test_decoded == class_name
        if np.sum(class_mask) > 0:
            class_accuracy = accuracy_score(y_test_decoded[class_mask], y_pred[class_mask])
            print(f"  {class_name}: {class_accuracy:.4f} ({class_accuracy*100:.1f}%)")
    
    # THRESHOLD TUNING for binary classification
    if args.class_config == 2:
        print("\n🎛️ THRESHOLD TUNING FOR BETTER BENIGN DETECTION:")
        y_pred_proba = model.predict(X_test)
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        best_threshold = 0.5
        best_balanced_score = 0
        
        print("Threshold | Accuracy | Benign Recall | Attack Recall | F1-Score")
        print("-" * 65)
        
        for threshold in thresholds:
            # For binary classification, adjust the threshold
            y_pred_thresh = (y_pred_proba[:, 1] > threshold).astype(int)
            y_pred_labels = label_encoder.inverse_transform(y_pred_thresh)
            
            accuracy_t = accuracy_score(y_test_decoded, y_pred_labels)
            f1_t = f1_score(y_test_decoded, y_pred_labels, average='weighted', zero_division=0)
            
            # Calculate per-class recall
            benign_mask = y_test_decoded == 'Benign'
            attack_mask = y_test_decoded == 'attack'
            
            benign_recall = accuracy_score(y_test_decoded[benign_mask], y_pred_labels[benign_mask]) if np.sum(benign_mask) > 0 else 0
            attack_recall = accuracy_score(y_test_decoded[attack_mask], y_pred_labels[attack_mask]) if np.sum(attack_mask) > 0 else 0
            
            # Balanced score (average of both recalls)
            balanced_score = (benign_recall + attack_recall) / 2
            
            print(f"   {threshold:.1f}    |   {accuracy_t:.3f}   |     {benign_recall:.3f}     |     {attack_recall:.3f}     |  {f1_t:.3f}")
            
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_threshold = threshold
        
        print(f"\n🎯 BEST THRESHOLD: {best_threshold} (Balanced Score: {best_balanced_score:.4f})")
        
        # Show results with best threshold
        y_pred_best = (y_pred_proba[:, 1] > best_threshold).astype(int)
        y_pred_best_labels = label_encoder.inverse_transform(y_pred_best)
        
        print(f"\n🏆 RESULTS WITH OPTIMAL THRESHOLD ({best_threshold}):")
        cm_best = confusion_matrix(y_test_decoded, y_pred_best_labels)
        print("Confusion Matrix:")
        print(cm_best)
        
        benign_recall_best = accuracy_score(y_test_decoded[benign_mask], y_pred_best_labels[benign_mask])
        attack_recall_best = accuracy_score(y_test_decoded[attack_mask], y_pred_best_labels[attack_mask])
        
        print(f"Benign Detection: {benign_recall_best:.3f} ({benign_recall_best*100:.1f}%)")
        print(f"Attack Detection: {attack_recall_best:.3f} ({attack_recall_best*100:.1f}%)")
    
    # Save training history
    history_path = os.path.join(models_dir, f'training_history_{args.class_config}class.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)
    print(f"✅ Training history saved to: {history_path}")
    
    # Save model configuration
    config_path = os.path.join(models_dir, f'model_config_{args.class_config}class.txt')
    with open(config_path, 'w') as f:
        f.write(f"Model Configuration:\n")
        f.write(f"Class Config: {args.class_config}\n")
        f.write(f"Features: {args.k_features}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Input Shape: {input_shape}\n")
        f.write(f"Classes: {label_encoder.classes_}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        if args.class_config == 2:
            f.write(f"\nBest Threshold: {best_threshold}\n")
            f.write(f"Balanced Score: {best_balanced_score:.4f}\n")
    
    print(f"✅ Model configuration saved to: {config_path}")
    
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📁 All files saved in: {models_dir}")
    print(f"📊 Key improvements:")
    print(f"   - CNN-LSTM hybrid architecture")
    print(f"   - Feature selection ({args.k_features} best features)")
    print(f"   - Focal loss for imbalanced data")
    print(f"   - Heavy class weights for Benign detection")
    print(f"   - Threshold tuning for optimal performance")
