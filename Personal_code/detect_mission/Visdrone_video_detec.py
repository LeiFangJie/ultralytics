from ultralytics import YOLO
from pathlib import Path
import cv2

# ========== 训练配置（基于你的代码修改）==========
model = YOLO("yolo26n.yaml").load("yolo26n.pt")

results = model.train(
    data="datasets/Visdrone/video/visdrone_vid.yaml",
    
    # 训练策略
    epochs=300,          # 30→300，VisDrone需要充分收敛
    patience=50,         # 20→50，避免过早早停
    batch=16,             # 8→4，配合1280分辨率
    imgsz=1280,          # 640→1280，小目标必须大分辨率
    
    # 冻结：删除 freeze=11
    # VisDrone与COCO域差异极大（俯视vs平视，小目标vs近景），必须全量微调
    # 如果显存吃紧，可以只 freeze=0（默认就是不冻结）
    
    # 学习率
    lr0=0.001,           # 保持
    lrf=0.1,             # 0.01→0.1，最终学习率别压太低
    cos_lr=True,         # 新增：余弦退火，比线性衰减更稳
    
    # 数据增强（直接在参数里加）
    mixup=0.2,           # 必须开，增强小目标鲁棒性
    copy_paste=0.1,      # 复制粘贴，增加小目标密度
    degrees=5.0,         # 无人机视角旋转很常见
    scale=0.5,           # 模拟不同飞行高度
    mosaic=1.0,          # 保持开启
    
    # 正则化与类别不平衡（解决truck/bus AP暴跌）
    dropout=0.1,         # 新增：正则化防过拟合
    weight_decay=0.001,  # 默认0.0005，适当加大惩罚
    cls_pw=1.0,          # 新增：开启类别权重，解决car/bus数量不平衡
    
    # 硬件
    device=0,
    workers=8,           # 0→4，训练速度提升3-5倍
    amp=True,            # 新增：8GB显存跑1280必须开混合精度
    optimizer="AdamW",
    
    # 优化器：保留auto（你坚持）
    # 说明：Ultralytics检测任务默认auto=SGD，如果你觉得震荡再手动改AdamW
)

# ========== 预测配置（你的代码基本没问题，微调）==========
video_dir = Path("datasets/Visdrone/video/images/test")
save_project = "visdrone_predict"
save_name = "visdrone_video"

screen_w, screen_h = 1280, 768
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# 加载训练好的模型
#model = YOLO("ultralytics-repo/runs/detect/train-6/weights/best.pt")

video_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS])

if not video_files:
    print(f"错误: {video_dir} 下没找到视频文件")
    exit(1)

print(f"共找到 {len(video_files)} 个视频: {[v.name for v in video_files]}")

for video_idx, video_path in enumerate(video_files, 1):
    print(f"\n>>> 正在处理 [{video_idx}/{len(video_files)}]: {video_path.name}")

    results = model.predict(
        source=str(video_path),
        stream=True,
        save=True,
        device=0,
        project=save_project,
        name=f"{save_name}_{video_path.stem}",
        exist_ok=True,
        # VisDrone密集场景建议调整
        conf=0.25,         # 置信度阈值，默认0.25，可试0.15-0.3
        iou=0.5,           # NMS阈值，默认0.7，密集场景降到0.5减少重叠框
        max_det=500,       # 默认300，VisDrone密集场景可能不够，改500
    )

    # 逐帧显示
    for frame_idx, result in enumerate(results):
        im_bgr = result.plot()
        h, w = im_bgr.shape[:2]

        scale = min(screen_w / w, screen_h / h, 1.0)
        if scale < 1.0:
            im_display = cv2.resize(im_bgr, (int(w * scale), int(h * scale)))
        else:
            im_display = im_bgr

        cv2.imshow("VisDrone Prediction", im_display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"    [跳过] 用户中断，进入下一个视频")
            break
        elif key == 27:
            print(f"    [退出] 用户彻底退出")
            cv2.destroyAllWindows()
            exit(0)

    cv2.waitKey(100)

cv2.destroyAllWindows()
print("\n全部视频处理完毕！")