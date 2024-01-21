from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    # model = YOLO('C:/Users/fengx/AppData/Local/Programs/Python/Python311/Lib/site-packages/ultralytics/cfg/models/v8/yolov8_SE.yaml')  # build a new model from scratch
    model = YOLO('G:/YoloTest/ultralytics/cfg/models/v8/yolov8_SE.yaml')
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('G:/YoloTest/ultralytics/cfg/models/v8/yolov8_SE.yaml').load('yolov8n.pt')
    # model = YOLO('C:/Users/fengx/AppData/Local/Programs/Python/Python311/Lib/site-packages/ultralytics/cfg/models/v8/yolov8_SE.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
    # Train the model
    results = model.train(data="lung.yaml",
                          epochs=50,
                          imgsz=1024,
                          batch=8)
    # validation
    metrics = model.val()
    # pred
    results = model("./dataset/val/images/00020438_011.png")
    success = model.export(format="onnx")