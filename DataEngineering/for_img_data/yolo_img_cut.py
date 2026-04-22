import os
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ================= 配置区域 =================
# 输入文件夹的根目录 (请修改为你自己的路径)
INPUT_ROOT = "/Users/fron/codes/CapstoneCUC/row_storage/img_data/LoRAs"

# 输出文件夹的根目录 (脚本会自动创建)
OUTPUT_ROOT = "/Users/fron/codes/CapstoneCUC/processed_dataset_512"

# 目标尺寸
TARGET_SIZE = 512

# 加载模型 (M4芯片会自动使用MPS加速，或者CPU也很快)
print("正在加载 YOLOv8 模型...")
model = YOLO('yolov8n.pt')
print("模型加载完成！")

# ================= 核心函数 =================

def smart_resize(img, target_size):
    """
    将图片按比例缩放，使得短边 = target_size
    这样可以保证后续裁剪时不会出现黑边
    """
    w, h = img.size
    if min(w, h) < target_size:
        # 如果图片太小，必须放大
        ratio = target_size / min(w, h)
    elif min(w, h) > target_size:
        # 如果图片太大，缩小以节省计算资源，同时突出主体
        ratio = target_size / min(w, h)
    else:
        return img

    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return img.resize((new_w, new_h), Image.LANCZOS)

def get_crop_box(img_w, img_h, center_x, center_y, size):
    """
    根据中心点计算裁剪框，并处理边界溢出
    """
    half = size // 2
    
    # 初步计算左上角 (x1, y1)
    x1 = int(center_x - half)
    y1 = int(center_y - half)
    
    # 边界修正：不能小于0，也不能超出图片范围
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    
    if x1 + size > img_w: x1 = img_w - size
    if y1 + size > img_h: y1 = img_h - size
    
    return (x1, y1, x1 + size, y1 + size)

def process_single_image(src_path, dst_path):
    try:
        # 1. 打开图片
        img = Image.open(src_path).convert('RGB')
        
        # 2. 智能缩放 (确保短边=512)
        img = smart_resize(img, TARGET_SIZE)
        
        # 3. 运行 YOLO 检测
        # 这里的 img 是 PIL 格式，YOLO可以直接吃
        results = model(img, verbose=False) # verbose=False 关闭每张图的打印刷屏
        
        # 4. 寻找“主角车”
        best_box = None
        max_area = 0
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                # COCO数据集: 2=car, 7=truck, 5=bus (有时候Model Y会被识别为SUV/Truck)
                if cls_id in [2, 5, 7]: 
                    # 计算面积，我们要最大的那辆
                    xyxy = box.xyxy[0].tolist()
                    w_box = xyxy[2] - xyxy[0]
                    h_box = xyxy[3] - xyxy[1]
                    area = w_box * h_box
                    
                    if area > max_area:
                        max_area = area
                        best_box = xyxy

        # 5. 决定裁剪中心
        img_w, img_h = img.size
        
        if best_box:
            # 方案A: 找到车了，以车为中心
            center_x = (best_box[0] + best_box[2]) / 2
            center_y = (best_box[1] + best_box[3]) / 2
            print(f"  [智能裁剪] 发现车辆，以 ({int(center_x)}, {int(center_y)}) 为中心")
        else:
            # 方案B: 没找到车 (可能是内饰或特写)，回退到图片中心
            center_x = img_w / 2
            center_y = img_h / 2
            print(f"  [中心裁剪] 未检测到车辆，使用图片中心")

        # 6. 执行裁剪
        crop_box = get_crop_box(img_w, img_h, center_x, center_y, TARGET_SIZE)
        cropped_img = img.crop(crop_box)
        
        # 7. 保存
        cropped_img.save(dst_path, quality=95)
        
    except Exception as e:
        print(f"  [错误] 处理失败 {src_path}: {e}")

# ================= 主程序 =================

def main():
    # 遍历 INPUT_ROOT 下的所有子文件夹
    for root, dirs, files in os.walk(INPUT_ROOT):
        for file in files:
            # 只处理图片文件
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                src_path = os.path.join(root, file)
                
                # 构建输出路径，保持文件夹结构
                # 例如: .../LoRAs/life/1.jpg -> .../processed/life/1.jpg
                relative_path = os.path.relpath(root, INPUT_ROOT)
                dst_folder = os.path.join(OUTPUT_ROOT, relative_path)
                
                if not os.path.exists(dst_folder):
                    os.makedirs(dst_folder)
                
                dst_path = os.path.join(dst_folder, file)
                
                print(f"正在处理: {relative_path}/{file}")
                process_single_image(src_path, dst_path)

    print("\n" + "="*30)
    print(f"全部完成！处理后的图片保存在: {OUTPUT_ROOT}")
    print("="*30)

if __name__ == "__main__":
    main()