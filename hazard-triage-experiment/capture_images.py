"""
Step 2 - Dataset capture for the Sample Hazard Triage experiment.

Controls (with the video window focused):
  h - save current frame as 'hazardous' (your red object)
  y - save current frame as 'rare'      (your yellow object)
  q - quit

Move, rotate, and change the distance/angle of the object between captures.
Vary lighting if you can. Aim for 150-300 images per class - variety matters
more than raw count.
"""
import cv2
import os
from datetime import datetime

DATASET_DIR = "dataset"
CLASSES = {
    ord('h'): "hazardous",  # red object
    ord('y'): "rare",       # yellow object
}


def ensure_dirs():
    for class_name in CLASSES.values():
        os.makedirs(os.path.join(DATASET_DIR, class_name), exist_ok=True)


def main():
    ensure_dirs()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Check Mac camera permissions.")
        return

    counts = {
        name: len(os.listdir(os.path.join(DATASET_DIR, name)))
        for name in CLASSES.values()
    }
    print("Starting counts:", counts)
    print("Press 'h' = save hazardous (red), 'y' = save rare (yellow), 'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        text = f"hazardous:{counts['hazardous']}  rare:{counts['rare']}"
        cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Capture - h/y to save, q to quit", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in CLASSES:
            class_name = CLASSES[key]
            counts[class_name] += 1
            filename = f"{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            path = os.path.join(DATASET_DIR, class_name, filename)
            cv2.imwrite(path, frame)
            print(f"Saved {path}  (total {class_name}: {counts[class_name]})")

    cap.release()
    cv2.destroyAllWindows()
    print("Final counts:", counts)


if __name__ == "__main__":
    main()
