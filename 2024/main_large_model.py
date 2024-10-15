from ultralytics import YOLO

# model = YOLO("configs/large_detection_cfg.yaml")  # load a pretrained model (recommended for training)
model = YOLO("configs/small_detection_cfg.yaml")  # load a pretrained model (recommended for training)

# Train the model
# results = model.train(data="keti_dataset.yaml", save_period=4, project='IITP', batch=-1, epochs=100, imgsz=640, plots=True)
results = model.train(data="configs/merge_keti_dataset.yaml", save_period=4, project='IITP', batch=-1, epochs=200, imgsz=640, plots=True)
