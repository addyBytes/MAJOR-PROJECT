import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
import keras_tuner as kt

def create_model(hp, input_shape, num_classes):
    """
    Builds and compiles a tunable hybrid CNN-LSTM model.
    The 'hp' argument is a Keras Tuner hyperparameter object.
    """
    model = Sequential()
    
    # --- Input Layer ---
    model.add(Input(shape=input_shape))
    
    # --- CNN Block ---
    model.add(Conv1D(
        filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32), 
        kernel_size=hp.Choice('cnn_kernel_size', [3, 5]), 
        activation='relu'
    ))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(
        rate=hp.Float('cnn_dropout', min_value=0.2, max_value=0.5, step=0.1)
    ))
    
    # --- LSTM Block ---
    model.add(LSTM(
        units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
        return_sequences=False 
    ))
    model.add(Dropout(
        rate=hp.Float('lstm_dropout', min_value=0.2, max_value=0.5, step=0.1)
    ))

    # --- Fully Connected (Dense) Block for Classification ---
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation='relu'
    ))
    
    # --- Output Layer ---
    model.add(Dense(num_classes, activation='softmax'))

    # --- Compile the Model ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [0.001, 0.0001])
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model