from ultralytics import YOLO
import os
from pathlib import Path
import cv2  # 新增

# 加载模型
model = YOLO("yolo26n.pt")

# 设置路径
video_folder = "datasets/Test_img_video"

video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

for video_file in sorted(os.listdir(video_folder)):
    if Path(video_file).suffix.lower() not in video_extensions:
        print(f'不支持的文件格式: {video_file}, 已跳过.')
        continue
    
    video_path = os.path.join(video_folder, video_file)
    print(f"正在处理: {video_file}")
    
    # 改动①：去掉 show=True，避免原生窗口超出屏幕
    results = model(video_path, save=True,
                      name=Path(video_file).stem, stream=True)
    
    for result in results:
        frame_id = result.path
        
        # 改动②：获取标注帧，缩小到屏幕能装下再显示
        annotated = result.plot()
        h, w = annotated.shape[:2]
        scale = min(1280 / w, 720 / h, 1.0)  # 最大显示 1280x720，保持比例
        if scale < 1.0:
            annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))
        cv2.imshow("Preview", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # 按 Q 提前结束当前视频
        
        if result.boxes is not None and len(result.boxes) > 0:
            xywh = result.boxes.xywh
            xyxy = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls
            names = [result.names[int(cls)] for cls in classes]
            print(f"  检测到 {len(names)} 个目标: {names}")
    
    cv2.destroyAllWindows()  # 改动③：关闭当前视频的预览窗口

print("全部处理完成！")