from ultralytics import YOLO

# model = YOLO("yolo26n.yaml").load("yolo26n.pt")

# results = model.train(
#     data="datasets/Visdrone/img/VisDrone.yaml",
#     epochs=10,          # 可以更长，配合早停
#     patience=20,         # 早停，防止过拟合
#     device=0,
#     workers=0,           # workers=0 训练慢，建议设 4-8
#     batch=8,
#     imgsz=960,           # 或尝试 960（如果显存够）
#     freeze=23,           # 冻结前 22 层只有detection head，保护预训练权重
#     lr0=0.001,           # 降低初始学习率，保护预训练权重
#     lrf=0.01,            # 最终学习率系数    
# )

model = YOLO("ultralytics-repo/runs/detect/train-32/weights/best.pt")  # 加载训练好的模型权重

source_path = "datasets/Visdrone/img/images/test"  # 你的img文件夹路径        

# 执行批量预测（stream=True 对大批量图片更省内存）
results = model.predict(
    source=source_path,      # 可以是文件夹、图片列表、视频、摄像头id
    conf=0.25,               # 置信度阈值，VisDrone小目标多可以适当调低（如0.15-0.2）
    iou=0.45,                # NMS IoU阈值
    imgsz=960,               # 推理尺寸，建议和训练一致
    device=0,                # GPU
    save=True,               # 保存带bbox的可视化图片
    project="visdrone_predict",  # 项目目录
    name="visdrone_img",   # 本次预测名称
    verbose=True,            # 打印每张图信息
    line_width=2,            # bbox线宽
)
