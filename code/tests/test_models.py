import numpy as np
from models.xception_lstm import build_model

def test_model_shape():
    model = build_model()
    dummy_input = np.random.rand(1, 224, 224, 3)
    output = model.predict(dummy_input)
    assert output.shape == (1, 1), "Model output shape incorrect"
    print("Model shape test passed.")

if __name__ == "__main__":
    test_model_shape()
