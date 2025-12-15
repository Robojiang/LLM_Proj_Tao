import pickle
import numpy as np
from tqdm import tqdm
import os

# --- 配置参数 ---
PROCESSED_DIR = './processed_data'
PRETRAINED_DIR = './pretrained_vectors' 

# 词汇表文件
ZH_VOCAB_FILE = os.path.join(PROCESSED_DIR, 'zh_vocab.pkl')
EN_VOCAB_FILE = os.path.join(PROCESSED_DIR, 'en_vocab.pkl')

# 预训练词向量文件
# 请确保这里指向的是解压后的 .txt 文件
EN_PRETRAINED_FILE = os.path.join(PRETRAINED_DIR, 'dolma_300_2024_1.2M.100_combined.txt')
ZH_PRETRAINED_FILE = os.path.join(PRETRAINED_DIR, 'sgns.baidubaike.bigram-char') 
# ZH_PRETRAINED_FILE = os.path.join(PRETRAINED_DIR, 'dolma_300_2024_1.2M.100_combined.txt') 

# 输出文件
EN_EMBEDDING_MATRIX_FILE = os.path.join(PROCESSED_DIR, 'en_embedding_matrix.npy')
ZH_EMBEDDING_MATRIX_FILE = os.path.join(PROCESSED_DIR, 'zh_embedding_matrix.npy')

# 默认维度 (会被文件头信息覆盖)
DEFAULT_ZH_DIM = 300 
DEFAULT_EN_DIM = 300

def build_matrix_optimized(word2id, pretrained_filepath, default_dim, is_zh=False):
    """
    内存优化版：流式读取大文件，只提取需要的词向量。
    避免将整个 10GB+ 的文件读入内存。
    """
    vocab_size = len(word2id)
    print(f"正在处理词汇表，大小: {vocab_size}")
    
    # 1. 先尝试读取第一行获取维度，或者使用默认维度
    embedding_dim = default_dim
    skip_first_line = False
    
    try:
        with open(pretrained_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip().split()
            if len(first_line) == 2:
                # 若第一行为: 词汇量 维度 (例如: 8824330 200)
                embedding_dim = int(first_line[1])
                skip_first_line = True
                print(f"检测到文件头，维度为: {embedding_dim}")
            else:
                # GloVe 没有文件头，第一行就是数据
                embedding_dim = len(first_line) - 1
                print(f"未检测到文件头，根据第一行推断维度为: {embedding_dim}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {pretrained_filepath}")
        return None

    # 2. 初始化随机矩阵 (正态分布)
    # scale=0.6 是经验值，让初始化的方差不要太大
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, embedding_dim))
    
    # 将 <PAD> 初始化为 0
    if '<PAD>' in word2id:
        embedding_matrix[word2id['<PAD>']] = np.zeros(embedding_dim)

    # 3. 流式读取文件
    print(f"正在扫描预训练文件 (流式读取): {pretrained_filepath} ...")
    hits = 0
    
    with open(pretrained_filepath, 'r', encoding='utf-8', errors='ignore') as f:
        if skip_first_line:
            f.readline() # 跳过第一行
            
        for line in tqdm(f):
            # 简单的 split 可能会比较慢，但对于标准格式足够了
            # 为了加速，可以只 split 第一个空格找到词
            line = line.rstrip()
            if not line: continue
            
            # 找到第一个空格的位置
            space_idx = line.find(' ')
            if space_idx == -1: continue
            
            word = line[:space_idx]
            
            # 关键点：只有当这个词在我们的词表里时，才处理后面的向量数据
            if word in word2id:
                idx = word2id[word]
                try:
                    # 取出向量部分并转换
                    vector_str = line[space_idx+1:]
                    # fromstring 比 split 后再转换要快
                    vector = np.fromstring(vector_str, sep=' ')
                    
                    # 再次检查维度是否匹配 (防止坏数据)
                    if len(vector) == embedding_dim:
                        embedding_matrix[idx] = vector
                        hits += 1
                except Exception:
                    continue

    print(f"构建完成。命中词数: {hits} / {vocab_size} (覆盖率: {hits/vocab_size:.2%})")
    return embedding_matrix

def main():
    # 确保目录存在
    if not os.path.exists(PRETRAINED_DIR):
        os.makedirs(PRETRAINED_DIR)
        print(f"请将下载的预训练词向量放入 {PRETRAINED_DIR}")
        return

    # --- 处理英文 ---
    if os.path.exists(EN_PRETRAINED_FILE):
        print("\n--- 开始构建英文嵌入矩阵 ---")
        with open(EN_VOCAB_FILE, 'rb') as f:
            en_word2id = pickle.load(f)
        
        en_matrix = build_matrix_optimized(en_word2id, EN_PRETRAINED_FILE, DEFAULT_EN_DIM, is_zh=False)
        if en_matrix is not None:
            np.save(EN_EMBEDDING_MATRIX_FILE, en_matrix)
            print(f"保存至: {EN_EMBEDDING_MATRIX_FILE}")
    else:
        print(f"\n警告: 未找到英文预训练文件 {EN_PRETRAINED_FILE}，跳过。")

    # --- 处理中文 ---
    if os.path.exists(ZH_PRETRAINED_FILE):
        print("\n--- 开始构建中文嵌入矩阵 ---")
        with open(ZH_VOCAB_FILE, 'rb') as f:
            zh_word2id = pickle.load(f)
        
        zh_matrix = build_matrix_optimized(zh_word2id, ZH_PRETRAINED_FILE, DEFAULT_ZH_DIM, is_zh=True)
        if zh_matrix is not None:
            np.save(ZH_EMBEDDING_MATRIX_FILE, zh_matrix)
            print(f"保存至: {ZH_EMBEDDING_MATRIX_FILE}")
    else:
        print(f"\n警告: 未找到中文预训练文件 {ZH_PRETRAINED_FILE}，跳过。")

if __name__ == '__main__':
    main()