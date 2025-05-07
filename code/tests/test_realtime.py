import cv2

def test_camera():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Camera not accessible"
    ret, frame = cap.read()
    assert ret, "Failed to read frame"
    cap.release()
    print("Camera test passed.")

if __name__ == "__main__":
    test_camera()
