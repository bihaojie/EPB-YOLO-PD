from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/exp569/weights/best.pt')
    model.val(data=r'dataset\data.yaml',
              split='test',
              imgsz=640,
              batch=4,
              project='runs/test',
              name=f'exp569',
              )