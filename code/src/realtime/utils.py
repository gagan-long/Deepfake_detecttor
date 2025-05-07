import cv2

def preprocess_frame(frame, size=(224, 224)):
    """Resize and normalize a frame for model input."""
    frame = cv2.resize(frame, size)
    frame = frame / 255.0
    return frame
