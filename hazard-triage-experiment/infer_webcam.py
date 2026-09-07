"""
Step 5 - Run your trained model live on the webcam.
Press 'q' to quit.
"""
from ultralytics import YOLO
import cv2

MODEL_PATH = "runs_hazard_triage/v1/weights/best.pt"


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Running live detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Hazard Triage - live detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
