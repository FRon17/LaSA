import os
import base64
import requests
import json
from tqdm import tqdm
import time

# ================= 配置区域 =================

# 1. 输入文件夹 (拼好的1024图片文件夹)
INPUT_ROOT = "/Users/fron/codes/CapstoneCUC/row_storage/img_data/training_dataset_1024"

# 2. 硅基流动 API Key (请替换为你自己的 Key)
API_KEY = "sk-exnfevuvrfvvsnihchhmzdgannmzsnmzvsekudhrjbafxfiv"  # <--- 替换这里！

# 3. API 地址 (硅基流动 Qwen-VL-Max)
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3.5-397B-A17B" # 或者 Qwen/Qwen2-VL-7B-Instruct (更便宜)

# 4. 风格定义 (根据文件夹名映射)
# 键名必须与你的文件夹名一致 (小写, 下划线)
STYLE_DEFINITIONS = {
    "mobile_living": {
        "name": "Mobile Living",
        "desc": "Cozy car interior, mattress in trunk, camping mode, view from inside car. Emphasize warmth, privacy, and lifestyle."
    },
    "urban_cinematic": {
        "name": "Urban Cinematic",
        "desc": "City street, night, neon lights, bokeh, cinematic lighting, masterpiece. Emphasize the reflection of city lights on the car body."
    },
    "dark_aesthetics": {
        "name": "Dark Aesthetics",
        "desc": "Matte black, all black, black rims, cool, mysterious. Emphasize the sleek, powerful, and stealthy look."
    },
    "minimalist_tech": {
        "name": "Minimalist Tech",
        "desc": "White car, minimalist, clean lines, futuristic, bright studio light. Emphasize purity, technology, and high-key lighting."
    },
    # 默认 fallback
    "default": {
        "name": "Tesla Model Y Style",
        "desc": "High quality, professional photography of Tesla Model Y."
    }
}

# ================= 核心函数 =================

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption(image_path, style_info):
    base64_image = encode_image(image_path)
    
    prompt = f"""
# Role
You are an expert AI visual captioner.

# Context
The images belong to the category: "{style_info['name']}".
Definition: "{style_info['desc']}"

# Task
Describe this 2x2 grid image of a Tesla Model Y.
Format: "[Global Description]; [IMAGE1] ..., [IMAGE2] ..., [IMAGE3] ..., and [IMAGE4] ..."

# Requirements
1. Global Description must mention "Tesla Model Y" and "{style_info['name']}".
2. Describe [IMAGE1] (Top-Left), [IMAGE2] (Top-Right), [IMAGE3] (Bottom-Left), [IMAGE4] (Bottom-Right).
3. Single paragraph, no line breaks.
4. Keep it concise but descriptive.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.7
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status() # 检查 HTTP 错误
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"API 调用失败: {e}")
        if 'response' in locals():
            print(response.text)
        return None

# ================= 主程序 =================

def main():
    # 遍历所有子文件夹
    for root, dirs, files in os.walk(INPUT_ROOT):
        # 确定当前文件夹的风格
        folder_name = os.path.basename(root).lower()
        style_info = STYLE_DEFINITIONS.get(folder_name, STYLE_DEFINITIONS["default"])
        
        # 获取所有图片
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not image_files:
            continue
            
        print(f"\n正在处理文件夹: {folder_name} (风格: {style_info['name']})")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(root, img_file)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            
            # 如果 txt 已存在，跳过 (断点续传)
            if os.path.exists(txt_path):
                continue
                
            caption = generate_caption(img_path, style_info)
            
            if caption:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                # 稍微延时一下，防止 API 限流 (根据你的配额调整)
                time.sleep(0.5)
            else:
                print(f"跳过图片: {img_file}")

    print("\n" + "="*30)
    print(f"全部完成！Caption 已保存在: {INPUT_ROOT}")
    print("="*30)

if __name__ == "__main__":
    main()