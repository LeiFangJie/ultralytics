from ultralytics import YOLO
import os
from pathlib import Path

# 加载模型
model = YOLO("path/to/best.pt")

# 设置路径
video_folder = "datasets/Test_img_video"  # 你的视频文件夹
output_folder = "test_output"      # 结果保存文件夹（可选）
os.makedirs(output_folder, exist_ok=True)

# 支持的视频格式
video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

# 遍历文件夹中的所有视频
for video_file in sorted(os.listdir(video_folder)):
    # 检查是否是视频文件
    if Path(video_file).suffix.lower() not in video_extensions:
        print(f'不支持的文件格式: {video_file}, 已跳过.')
        continue
    
    video_path = os.path.join(video_folder, video_file)
    print(f"正在处理: {video_file}")
    
    # 直接对视频进行推理（YOLO 会自动逐帧处理）
    results = model(video_path, save=True, project=output_folder, name=Path(video_file).stem, stream=True)
    
    # 如果你想手动处理每一帧的结果：
    for result in results:
        # 当前是第几帧
        frame_id = result.path  # 当前帧信息
        
        # 提取检测框信息
        if result.boxes is not None and len(result.boxes) > 0:
            xywh = result.boxes.xywh      # 中心坐标 + 宽高
            xyxy = result.boxes.xyxy      # 左上角 + 右下角
            confs = result.boxes.conf     # 置信度
            classes = result.boxes.cls    # 类别索引
            
            # 获取类别名称
            names = [result.names[int(cls)] for cls in classes]
            
            # 在这里你可以做自定义操作，比如：
            # - 统计每帧的目标数量
            # - 根据置信度过滤
            # - 记录到文件
            print(f"  检测到 {len(names)} 个目标: {names}")

print("全部处理完成！")