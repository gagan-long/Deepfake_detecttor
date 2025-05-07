import cv2
import numpy as np
from models.xception_lstm import load_model

class RealTimeDetector:
    def __init__(self):
        self.model = load_model('models/weights.h5')
        self.cap = cv2.VideoCapture(0)
        
    def process_frame(self, frame):
        resized = cv2.resize(frame, (224, 224))
        prediction = self.model.predict(np.expand_dims(resized, 0))
        return prediction[0][0]
    
    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret: break
            
            confidence = self.process_frame(frame)
            label = f"Fake: {confidence:.2f}" if confidence > 0.5 else "Real"
            
            cv2.putText(frame, label, (10,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Deepfake Detector', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
