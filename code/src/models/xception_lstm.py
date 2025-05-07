from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense

def build_model():
    base = tf.keras.applications.Xception(
        include_top=False, 
        input_shape=(224,224,3)
    )
    x = base.output
    x = Reshape((-1, 2048))(x)
    x = LSTM(128)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)
