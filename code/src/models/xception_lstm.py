import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras import Model

def build_model():
    base = tf.keras.applications.Xception(
        include_top=False, 
        input_shape=(224,224,3),
        pooling='avg'
    )
    x = base.output
    x = Reshape((1, -1))(x)
    x = LSTM(128)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def load_model(weights_path):
    model = build_model()
    model.load_weights(weights_path)
    return model
