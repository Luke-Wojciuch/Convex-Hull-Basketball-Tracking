import torch
from ultralytics import YOLO

def main():
    # Ensure that the device is set to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Load the model
    model = YOLO('yolov8n.pt')

    # Start training
    model.train(
        data='Model Data/data.yaml',
        batch=16,
        epochs=100,
        imgsz=640,
        device=device,
        patience=5,
        save=True,
        project='Model_Output',
        name='yolov8n_training'
    )

if __name__ == "__main__":
    main()
