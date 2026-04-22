# 负责批量将裁剪好的512图片按照 CLIP 特征相似度拼成1024的网格图，适用于 Tesla 图片数据集的整理
import os
import torch
from PIL import Image
import clip  # 使用 OpenAI 官方库
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 输入文件夹 (上一步裁剪好的512图片文件夹)
INPUT_ROOT = "./img_data/processed_dataset_512"

# 2. 输出文件夹 (最终用于训练的拼图)
OUTPUT_ROOT = "./img_data/training_dataset_1024"

# 3. 本地 CLIP 模型路径 (你提供的路径)
LOCAL_CLIP_PATH = "./CLIP/ViT-B-32.pt"

# 拼图设置
GRID_SIZE = 1024
CELL_SIZE = 512

# 设备选择
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"正在使用设备: {device}")

# ================= 加载本地 CLIP 模型 =================
print(f"正在加载本地 CLIP 模型: {LOCAL_CLIP_PATH} ...")
try:
    # jit=False 可以避免一些兼容性问题
    model, preprocess = clip.load(LOCAL_CLIP_PATH, device=device, jit=False)
    print("模型加载成功！")
except Exception as e:
    print(f"错误：无法加载模型。请确认路径正确且安装了 clip 库。\n{e}")
    exit()

# ================= 核心函数 =================

def get_image_embeddings(image_paths):
    """
    使用本地 CLIP 提取特征
    """
    embeddings = []
    batch_size = 16
    
    # 预处理所有图片
    images_tensor = []
    valid_paths = []
    
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            # 使用 clip 自带的预处理 (Resize, CenterCrop, Normalize)
            img_input = preprocess(img).unsqueeze(0)
            images_tensor.append(img_input)
            valid_paths.append(p)
        except Exception as e:
            print(f"无法读取图片 {p}: {e}")
    
    if not images_tensor:
        return None, []

    # 拼接成一个大 Tensor [N, 3, 224, 224]
    images_tensor = torch.cat(images_tensor).to(device)
    
    # 分批次计算特征
    print(f"正在计算 {len(images_tensor)} 张图片的特征向量...")
    with torch.no_grad():
        for i in range(0, len(images_tensor), batch_size):
            batch = images_tensor[i : i + batch_size]
            features = model.encode_image(batch)
            
            # 归一化 (重要！否则余弦相似度计算不准)
            features /= features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu())
            
    if not embeddings:
        return None, []
        
    return torch.cat(embeddings), valid_paths

def create_grid(images, save_path):
    """
    将 4 张 PIL 图片拼成 2x2 网格
    """
    grid_img = Image.new('RGB', (GRID_SIZE, GRID_SIZE))
    
    positions = [
        (0, 0), (CELL_SIZE, 0),
        (0, CELL_SIZE), (CELL_SIZE, CELL_SIZE)
    ]
    
    for i, img in enumerate(images):
        if i >= 4: break
        # 这里需要重新 Resize 到 512，因为 CLIP 的 preprocess 可能会把图缩放到 224
        if img.size != (CELL_SIZE, CELL_SIZE):
            img = img.resize((CELL_SIZE, CELL_SIZE))
        grid_img.paste(img, positions[i])
        
    grid_img.save(save_path, quality=95)

def process_folder(folder_path, output_folder):
    # 1. 获取图片路径
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp'))]
    image_paths = [os.path.join(folder_path, f) for f in image_files]
    
    if len(image_paths) < 4:
        print(f"文件夹 {folder_path} 图片少于4张，跳过。")
        return

    # 2. 计算特征向量
    embeddings, valid_paths = get_image_embeddings(image_paths)
    
    if embeddings is None:
        return

    # 3. 贪心分组算法 (逻辑不变)
    print("正在计算相似度矩阵并分组...")
    sim_matrix = torch.mm(embeddings, embeddings.t())
    
    used_indices = set()
    groups = []
    
    for i in range(len(valid_paths)):
        if i in used_indices:
            continue
            
        current_group = [i]
        used_indices.add(i)
        
        sim_scores = sim_matrix[i].clone()
        for used_idx in used_indices:
            sim_scores[used_idx] = -1.0
            
        if len(valid_paths) - len(used_indices) < 3:
            break
            
        _, top_indices = torch.topk(sim_scores, k=3)
        
        for idx in top_indices:
            idx = idx.item()
            current_group.append(idx)
            used_indices.add(idx)
            
        groups.append(current_group)

    print(f"成功组成 {len(groups)} 组拼图。")

    # 4. 执行拼图
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for group_idx, indices in enumerate(tqdm(groups, desc="Stitching")):
        images_to_stitch = []
        for idx in indices:
            # 重新读取原图进行拼接 (保证清晰度，CLIP预处理过的图太小了)
            img = Image.open(valid_paths[idx]).convert("RGB")
            images_to_stitch.append(img)
            
        save_name = f"grid_{group_idx:03d}.jpg"
        save_path = os.path.join(output_folder, save_name)
        create_grid(images_to_stitch, save_path)

# ================= 主程序 =================

def main():
    for root, dirs, files in os.walk(INPUT_ROOT):
        has_images = any(f.lower().endswith(('.jpg', '.png')) for f in files)
        
        if has_images:
            relative_path = os.path.relpath(root, INPUT_ROOT)
            output_folder = os.path.join(OUTPUT_ROOT, relative_path)
            
            print(f"\n正在处理文件夹: {relative_path}")
            process_folder(root, output_folder)

    print("\n" + "="*30)
    print(f"全部完成！拼图保存在: {OUTPUT_ROOT}")
    print("="*30)

if __name__ == "__main__":
    main()
