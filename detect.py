from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp569/weights/best.pt')
    model.predict(source='detect_images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp_569',
                  save=True,
                  conf=0.2,
                  iou=0.7,
                )
