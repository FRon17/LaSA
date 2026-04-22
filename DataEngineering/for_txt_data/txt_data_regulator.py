# 负责对小红书的文本数据进行正则化清洗，主要包括以下步骤：
# 1. 读取 Excel 文件，获取原始数据。
# 2. 将标题、描述和标签字段合并成一个新的字段 raw_content。
# 3. 定义一个清洗函数 clean_text，使用正则表达式和 emoji 库对文本进行清洗，去除 URL、@用户、小红书表情符、Emoji，以及非中文、英文、数字的字符。
# 4. 应用清洗函数到 raw_content 字段，生成 clean_content 字段。
# 5. 剔除清洗后为空的行，保存最终结果到新的 Excel 文件 cleaned_tesla_data.xlsx 中，只保留搜索词、clean_content、点赞数和收藏数字段。

import pandas as pd
import re
import emoji

# 1. 读取 Excel 文件
file_path = './txt_data/rednote_txt.xlsx' 
df = pd.read_excel(file_path)

print(f"原始数据行数: {len(df)}")

# 2. 字段合并
# 注意：有些字段可能是空的（NaN），需要先填充为空字符串，否则拼接会报错
df['标题'] = df['标题'].fillna('')
df['描述'] = df['描述'].fillna('')
df['标签'] = df['标签'].fillna('')

# 创建一个新列 'raw_content'，中间用空格隔开，防止词粘连
df['raw_content'] = df['标题'] + ' ' + df['描述'] + ' ' + df['标签']

# 3. 定义正则化清洗函数
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # (1) 转小写 (Model Y -> model y)
    text = text.lower()
    
    # (2) 去除 URL 链接 (http/https 开头)
    text = re.sub(r'http\S+', '', text)
    
    # (3) 去除 @用户 (例如 @特斯拉) - 视情况保留，通常@后面是人名，建议去除
    text = re.sub(r'@\S+', '', text)
    
    # (4) 去除小红书特有的表情符号 (例如 [笑哭] [R] [派对R])
    text = re.sub(r'\[.*?\]', '', text)
    
    # (5) 去除 Emoji (🚗, ✨) - 使用 emoji 库
    text = emoji.replace_emoji(text, replace='')
    
    # (6) 核心正则：只保留 中文、英文、数字
    # \u4e00-\u9fa5 代表汉字
    # a-z0-9 代表英文和数字
    # 其他所有符号（包括标点、#号、空格、换行符）都会被替换为空格
    text = re.sub(r'[^a-z0-9\u4e00-\u9fa5]', ' ', text)
    
    # (7) 去除多余的空格 (将多个空格合并为一个)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 4. 应用清洗函数
print("正在进行正则化清洗，请稍候...")
df['clean_content'] = df['raw_content'].apply(clean_text)

# 5. 简单预览清洗结果
print("\n--- 清洗前 (前3条) ---")
print(df['raw_content'].head(3).values)
print("\n--- 清洗后 (前3条) ---")
print(df['clean_content'].head(3).values)

# 6. 剔除清洗后为空的行 (有些帖子可能只有表情包)
df = df[df['clean_content'] != '']
print(f"清洗后剩余行数: {len(df)}")

# 7. 保存结果到新的 Excel
output_file = 'cleaned_tesla_data.xlsx'
# 只保留需要的字段，减小文件体积
df[['搜索词', 'clean_content', '点赞数', '收藏数']].to_excel(output_file, index=False)

print(f"\n处理完成！已保存为 {output_file}")
