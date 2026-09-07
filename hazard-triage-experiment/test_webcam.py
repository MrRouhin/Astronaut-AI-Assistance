"""
Step 1 - Run this first.
Just confirms your webcam opens and shows a live feed. Press 'q' to quit.
"""
import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        print("On Mac: check System Settings > Privacy & Security > Camera,")
        print("and make sure Terminal/VS Code has camera access.")
        return

    print("Webcam opened. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam Test - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
