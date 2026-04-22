import os

FOLDER_PATH = "/Users/fron/codes/CapstoneCUC/row_storage/img_data/training_dataset_1024"

replacements = {
    "the top-left image": "[IMAGE1]",
    "the top-right image": "[IMAGE2]",
    "the bottom-left image": "[IMAGE3]",
    "the bottom-right image": "[IMAGE4]",
    "top-left image": "[IMAGE1]", # 防止漏掉 'the'
    "top-right image": "[IMAGE2]",
    "bottom-left image": "[IMAGE3]",
    "bottom-right image": "[IMAGE4]",
    "the top-left": "[IMAGE1]",
    "the top-right": "[IMAGE2]",
    "the bottom-left": "[IMAGE3]",
    "the bottom-right": "[IMAGE4]"
}

count = 0
for root, dirs, files in os.walk(FOLDER_PATH):
    for file in files:
        if file.endswith(".txt"):
            path = os.path.join(root, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            new_content = content
            # 检查是否包含自然语言描述，如果有则替换
            for k, v in replacements.items():
                # 使用 case-insensitive 替换会更稳健，这里简单处理
                new_content = new_content.replace(k, v)
                new_content = new_content.replace(k.capitalize(), v) # 处理 The top-left...

            if new_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                count += 1
                print(f"已修复: {file}")

print(f"修复完成，共修改了 {count} 个文件。")