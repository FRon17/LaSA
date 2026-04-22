# 负责对清洗后的文本数据进行分词处理，使用 jieba 库进行中文分词，并结合停用词表过滤掉无意义的词汇。分词结果将保存到新的 Excel 文件中，方便后续的词频统计和分析。
import pandas as pd
import jieba

# =================配置区域=================
# 1. 输入文件路径 (上一以生成的清洗后的文件)
input_file = '/Users/fron/codes/CapstoneCUC/row_storage/txt_data/cleaned_tesla_data.xlsx' 

# 2. 停用词表路径 (你提供的路径)
stopwords_path = '/Users/fron/codes/CapstoneCUC/code_scripts/for_lora_b/哈工大停用词表.txt'

# 3. 输出文件路径
output_file = '/Users/fron/codes/CapstoneCUC/row_storage/txt_data/segmented_tesla_data.xlsx'
# =========================================

# --- 第一步：加载数据与停用词 ---
print("正在加载数据和停用词...")
df = pd.read_excel(input_file)

# 读取停用词表，转为集合(set)以提高查询速度
stopwords = set()
try:
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
except FileNotFoundError:
    print(f"错误：找不到停用词文件 {stopwords_path}")
    exit()

# --- 第二步：添加特斯拉/汽车领域专用词 (临时词典) ---
# 这一步非常重要！防止专有名词被切碎
# 这里的词会被强制识别为一个整体
custom_words = [
    "modely", "model3", "modelx", "models", # 车型
    "毛豆y", "毛豆3", "特斯", "特斯拉", # 昵称
    "黑武士", "黑化", "纳多灰", "冷光银", "深海蓝", "中国红", # 颜色/风格
    "极简风", "冷淡风", "科技感", "未来感", "赛博朋克", # 风格
    "单踏板", "动能回收", "哨兵模式", "露营模式", # 功能
    "零重力", "全景天幕", "大轮毂", "卡钳" # 硬件
]

for word in custom_words:
    jieba.add_word(word)

print(f"已添加 {len(custom_words)} 个领域专用词到词典。")

# --- 第三步：定义分词函数 ---
def cut_text(text):
    if not isinstance(text, str):
        return ""
    
    # 使用 jieba 精确模式分词
    words = jieba.lcut(text)
    
    result = []
    for word in words:
        # 1. 去除空格
        word = word.strip()
        
        # 2. 过滤逻辑：
        # (a) 必须不在停用词表中
        # (b) 长度必须大于1 (去除 '车', '看', '买' 等单字噪声)
        #     注意：如果你想保留 'y' 这种单字母，可以修改这里的逻辑，但在Model Y语境下通常不需要单独的y
        if word not in stopwords and len(word) > 1:
            result.append(word)
            
    # 将分词结果用空格连接，方便后续处理
    return " ".join(result)

# --- 第四步：执行分词 ---
print("正在进行分词，这可能需要一点时间...")
# 确保处理的是字符串类型
df['clean_content'] = df['clean_content'].astype(str)
df['segmented_text'] = df['clean_content'].apply(cut_text)

# --- 第五步：预览与保存 ---
print("\n--- 分词结果预览 (前5条) ---")
print(df['segmented_text'].head(5).values)

# 保存包含分词结果的新文件
df.to_excel(output_file, index=False)
print(f"\n分词完成！结果已保存至: {output_file}")

# --- 额外福利：打印最高频的20个词看看效果 ---
from collections import Counter
all_words = " ".join(df['segmented_text'].tolist()).split()
word_counts = Counter(all_words)
print("\n--- 当前最高频的20个词 ---")
print(word_counts.most_common(20))