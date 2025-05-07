import cv2
import numpy as np
from models.xception_lstm import load_model
from realtime.utils import preprocess_frame

def analyze_video(video_path):
    """Analyze a video file and return average fake confidence."""
    model = load_model('models/weights.h5')
    cap = cv2.VideoCapture(video_path)
    confidences = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 10 == 0:  # Sample every 10th frame for speed
            processed = preprocess_frame(frame)
            pred = model.predict(np.expand_dims(processed, 0))[0][0]
            confidences.append(pred)
        frame_count += 1

    cap.release()
    avg_conf = np.mean(confidences) if confidences else 0.0
    return {"confidence": avg_conf}
