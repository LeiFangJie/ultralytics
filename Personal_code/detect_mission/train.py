from ultralytics import YOLO

model = YOLO("yolo26n.yaml").load("yolo26n.pt")

results = model.train(
    data="datasets/kitti/kitti.yaml",
    epochs=100,          # 可以更长，配合早停
    patience=20,         # 早停，防止过拟合
    device=0,
    workers=0,           # workers=0 训练慢，建议设 4-8
    batch=16,
    imgsz=1280,           # 或尝试 1280（如果显存够）
    freeze=11,           # 冻结前 11 层（backbone 主干）
    lr0=0.001,           # 降低初始学习率，保护预训练权重
    lrf=0.01,            # 最终学习率系数
    close_mosaic=10,     # 最后 10 个 epoch 关掉 mosaic，稳定 BN
)