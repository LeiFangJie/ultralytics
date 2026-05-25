from ultralytics import YOLO
from pathlib import Path
import cv2

# model = YOLO("yolo26n.yaml").load("yolo26n.pt")

# results = model.train(
#     data="datasets/Visdrone/video/visdrone_vid.yaml",
#     epochs=30,          # 可以更长，配合早停
#     patience=20,         # 早停，防止过拟合
#     device=0,
#     workers=0,           # workers=0 训练慢，建议设 4-8
#     batch=8,
#     imgsz=640,           # 或尝试 960（如果显存够）
#     freeze=11,           # 冻结前 11 层只有detection head，保护预训练权重
#     lr0=0.001,           # 降低初始学习率，保护预训练权重
#     lrf=0.01,            # 最终学习率系数    
# )

# ========== 配置 ==========
video_dir = Path("datasets/Visdrone/video/images/test")  # 放多个视频的文件夹
save_project = "visdrone_predict"
save_name = "visdrone_video"

# 屏幕分辨率（用于4K视频自适应缩放）
screen_w, screen_h = 1280, 768

# 支持的视频格式
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

# ========== 加载模型 ==========
model = YOLO("ultralytics-repo/runs/detect/train-6/weights/best.pt")
# ========== 获取所有视频文件 ==========
video_files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS])

if not video_files:
    print(f"错误: {video_dir} 下没找到视频文件，请检查路径")
    exit(1)

print(f"共找到 {len(video_files)} 个视频: {[v.name for v in video_files]}")

# ========== 逐个视频处理 ==========
for video_idx, video_path in enumerate(video_files, 1):
    print(f"\n>>> 正在处理 [{video_idx}/{len(video_files)}]: {video_path.name}")

    # 每个视频单独调用 predict，stream=True 逐帧处理不爆内存
    results = model.predict(
        source=str(video_path),    # ✅ 传入单个视频路径，而不是整个文件夹
        stream=True,               # 必须 True，逐帧返回
        save=True,                 # 保存带bbox的视频/图片
        device=0,
        project=save_project,
        name=f"{save_name}_{video_path.stem}",  # 每个视频单独一个子目录
        exist_ok=True
    )

    # 内层循环：逐帧显示
    for frame_idx, result in enumerate(results):
        im_bgr = result.plot()  # 带bbox的BGR帧
        h, w = im_bgr.shape[:2]

        # 自适应缩放：4K视频缩放到屏幕内
        scale = min(screen_w / w, screen_h / h, 1.0)
        if scale < 1.0:
            im_display = cv2.resize(im_bgr, (int(w * scale), int(h * scale)))
        else:
            im_display = im_bgr

        cv2.imshow("VisDrone Prediction", im_display)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print(f"    [跳过] 用户中断，进入下一个视频")
            break  # ✅ 只跳出内层循环，外层继续下一个视频
            
        elif key == 27:  # ESC键
            print(f"    [退出] 用户彻底退出")
            cv2.destroyAllWindows()
            exit(0)

    # 每个视频处理完后，稍微停顿一下再加载下一个
    cv2.waitKey(100)

cv2.destroyAllWindows()
print("\n全部视频处理完毕！")