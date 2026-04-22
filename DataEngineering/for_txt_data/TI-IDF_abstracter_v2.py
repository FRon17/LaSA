import pandas as pd
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 读取数据
input_file = './segmented_tesla_data.xlsx'
df = pd.read_excel(input_file)

# 2. 【升级版】强力清洗黑名单
# 这次我们把“人”和“抽象概念”都杀掉，只留“物”和“景”
waste_words = {
    # 品牌/车型 (老生常谈)
    '特斯拉', 'model', 'modely', 'model3', '毛豆y', '汽车', '电车', '新能源', '新款', '新版',
    # 功能/参数/抽象 (无法画出)
    '续航', '升级', '模式', '驾驶', '功能', '系统', '电池', '自动驾驶', '加速', '动力', '安全', '车机',
    '体验', '喜欢', '感觉', '觉得', '直接', '没有', '心动', '感受', '推荐', '出发', '生活', '落地',
    '时候', '真的', '就是', '可以', '我们', '大家', '视频', '日常', '分享', '车主', '提车', '作业',
    '记录', '大片', '颜值', '好看', '姐妹', '女生', '男孩', '男人', '女人', '老公', '老婆', # 去掉人物
    '壁纸', '高清', '拍照', '摄影', '拍摄', '原图', '参数', '多少', '价格', '费用', # 去掉摄影/价格元数据
    # 其他常见废词
    '氛围','手机','空间','新车','朋友','有点','毛豆','落地','价格','优惠','活动','颜色','全程','买车','补贴',
    '政策','时间','后轮','后备箱','床垫','车型','车载','个人','小米','小时','油车',"新疆",'上海','官方','座椅',
    '地方','回家','地方','姐妹','后排','销售','理想','用车','人们','特斯','计划','二手车','问题','宝子','成都',
    '完全'
}

# 3. 定义筛选函数
def clean_and_filter(text):
    if not isinstance(text, str):
        return ""
    
    words = pseg.cut(text)
    result = []
    
    for word, flag in words:
        # 1. 剔除黑名单
        if word in waste_words:
            continue
        # 2. 剔除单字
        if len(word) < 2:
            continue
        # 3. 只保留：名词(n), 形容词(a), 动名词(vn)
        # 这一次我们放宽一点点，把 vn (名动词) 也加进来，防止漏掉 '改装' '贴膜' 这种词
        if flag.startswith('n') or flag.startswith('a') or flag.startswith('vn'):
            result.append(word)
            
    return " ".join(result)

print("正在执行深层挖掘清洗...")
df['final_keywords_deep'] = df['clean_content'].apply(clean_and_filter)

# 4. 重新计算 TF-IDF (扩大范围)
print("正在提取 Top 100 标签...")
# max_features=100: 我们要看更多词
# min_df=3: 哪怕只出现了3次，只要权重高，也捞出来看看
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=3, max_features=100) 

tfidf_matrix = tfidf_vectorizer.fit_transform(df['final_keywords_deep'])
feature_names = tfidf_vectorizer.get_feature_names_out()
weights = tfidf_matrix.mean(axis=0).A1
word_weights = dict(zip(feature_names, weights))
sorted_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)

# 5. 输出结果
print("\n--- 💎 深挖出的 Top 50 视觉潜力词 ---")
# 我们打印前50个，足够你挑选8个了
for i, (word, weight) in enumerate(sorted_words[:50]):
    print(f"{i+1}. {word}: {weight:.4f}")

# 顺便保存一下，方便你复制到论文附录
pd.DataFrame(sorted_words, columns=['关键词', '权重']).to_excel('tesla_deep_keywords.xlsx', index=False)
print("\n完整列表已保存为 tesla_deep_keywords.xlsx")
