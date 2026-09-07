"""
Step 4 - Train the detector. Run this AFTER you've labeled your images
(see the labeling note below) and exported them in YOLOv8 format.

Update DATA_YAML to point at the data.yaml file your labeling tool exported.
"""
from ultralytics import YOLO

DATA_YAML = "dataset_labeled/data.yaml"  # path from your Roboflow/LabelImg export
EPOCHS = 50
IMG_SIZE = 640


def main():
    model = YOLO("yolov8n.pt")  # starts from the pretrained nano model

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device="mps",   # Apple Silicon GPU. If this errors, change to "cpu"
        project="runs_hazard_triage",
        name="v1",
    )


if __name__ == "__main__":
    main()
