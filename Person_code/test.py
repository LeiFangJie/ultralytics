from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="datasets/kitti/kitti.yaml", epochs=50,
                      device=0, workers= 0,batch=32
                      )

