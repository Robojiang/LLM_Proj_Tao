import os
import json
import re
from collections import Counter
import pickle
import jieba
import hanlp
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from tqdm import tqdm



HanLP_Tokenizer = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH) 
# 或者使用更小的模型以加快速度: hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH
# 或者使用更准的大模型: hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH


# --- 配置参数 ---
# 数据目录
DATA_DIR = './data'
# 处理后数据的保存目录
PROCESSED_DIR = './processed_data'
# 训练文件名 (建议将所有训练数据放在一个文件里进行处理，或在此列表添加)
# 根据您的描述，这里包含了大小两个训练集
# TRAIN_FILES = ['train_100k.jsonl', 'train_10k.jsonl']
TRAIN_FILES = ['train_100k.jsonl']
# 验证和测试文件名
VALID_FILE = 'valid.jsonl'
TEST_FILE = 'test.jsonl'

# 句子最大长度
MAX_LEN = 80
# 词汇表最小词频
MIN_FREQ = 2

# --- 数据清理与分词 ---

def clean_str(string, lang):
    """
    字符串清理
    :param string: 输入字符串
    :param lang: 'zh' 或 'en'
    :return: 清理后的字符串
    """
    string = string.lower().strip()
    if lang == 'zh':
    # 保留：汉字、中文标点、0-9、常用英文符号、空格
        string = re.sub(r"[^\u4e00-\u9fa5，。！？?、；：“”‘’（）《》0-9A-Za-z\s%‰℃°$€£¥·@#&+=\-._/]", "", string)
    elif lang == 'en':
        string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def tokenize_en(sentence):
    """英文分词"""
    return word_tokenize(sentence)

# def tokenize_zh(sentence):
#     """中文分词"""
#     return jieba.lcut(sentence)

def tokenize_zh(sentence):
    """中文分词 (使用 HanLP)"""
    # HanLP 返回的是一个列表
    # 注意：HanLP 处理空字符串可能会报错或返回空，加个判断更稳健
    if not sentence.strip():
        return []
    return HanLP_Tokenizer(sentence)

def process_file(filepath, zh_vocab_counter, en_vocab_counter, is_train=False):
    """
    处理单个数据文件。
    如果是训练文件，则更新词汇表计数器。
    """
    print(f"正在处理文件: {filepath}")
    processed_pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing lines", unit="line"):
            pair = json.loads(line)
            zh_sent = pair.get('zh', '')
            en_sent = pair.get('en', '')

            # 1. 清理
            zh_sent_clean = clean_str(zh_sent, 'zh')
            en_sent_clean = clean_str(en_sent, 'en')

            # 2. 分词
            zh_tokens = tokenize_zh(zh_sent_clean)
            en_tokens = tokenize_en(en_sent_clean)

            # 3. 过滤过长句子
            if 0 < len(zh_tokens) <= MAX_LEN and 0 < len(en_tokens) <= MAX_LEN:
                processed_pairs.append({'zh_tokens': zh_tokens, 'en_tokens': en_tokens})
                # 4. 如果是训练集，则构建词汇表
                if is_train:
                    zh_vocab_counter.update(zh_tokens)
                    en_vocab_counter.update(en_tokens)
    return processed_pairs

def build_vocab(counter, min_freq):
    """根据计数器和最小词频构建词汇表"""
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    word_idx = len(vocab)
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = word_idx
            word_idx += 1
    return vocab

def save_data(data, filepath):
    """将处理后的数据保存为jsonl文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def save_vocab(vocab, filepath):
    """使用pickle保存词汇表"""
    with open(filepath, 'wb') as f:
        pickle.dump(vocab, f)

def main():
    """主执行函数"""
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # --- 1. 处理训练数据并构建词汇表 ---
    zh_vocab_counter = Counter()
    en_vocab_counter = Counter()
    
    all_train_data = []
    for filename in TRAIN_FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            train_data = process_file(filepath, zh_vocab_counter, en_vocab_counter, is_train=True)
            all_train_data.extend(train_data)
        else:
            print(f"警告: 训练文件 {filepath} 未找到。")

    save_data(all_train_data, os.path.join(PROCESSED_DIR, 'train.jsonl'))
    print(f"处理后的训练数据已保存到 {os.path.join(PROCESSED_DIR, 'train.jsonl')}")

    # --- 2. 构建并保存词汇表 ---
    zh_vocab = build_vocab(zh_vocab_counter, MIN_FREQ)
    en_vocab = build_vocab(en_vocab_counter, MIN_FREQ)

    save_vocab(zh_vocab, os.path.join(PROCESSED_DIR, 'zh_vocab.pkl'))
    save_vocab(en_vocab, os.path.join(PROCESSED_DIR, 'en_vocab.pkl'))
    print(f"中文词汇表大小: {len(zh_vocab)}")
    print(f"英文词汇表大小: {len(en_vocab)}")
    print(f"词汇表已保存到 {PROCESSED_DIR}")

    # --- 3. 处理验证和测试数据 ---
    # 注意：处理验证/测试集时不更新词汇表
    dummy_counter = Counter() # 空计数器
    
    valid_filepath = os.path.join(DATA_DIR, VALID_FILE)
    if os.path.exists(valid_filepath):
        valid_data = process_file(valid_filepath, dummy_counter, dummy_counter, is_train=False)
        save_data(valid_data, os.path.join(PROCESSED_DIR, 'valid.jsonl'))
        print(f"处理后的验证数据已保存到 {os.path.join(PROCESSED_DIR, 'valid.jsonl')}")
    else:
        print(f"警告: 验证文件 {valid_filepath} 未找到。")

    test_filepath = os.path.join(DATA_DIR, TEST_FILE)
    if os.path.exists(test_filepath):
        test_data = process_file(test_filepath, dummy_counter, dummy_counter, is_train=False)
        save_data(test_data, os.path.join(PROCESSED_DIR, 'test.jsonl'))
        print(f"处理后的测试数据已保存到 {os.path.join(PROCESSED_DIR, 'test.jsonl')}")
    else:
        print(f"警告: 测试文件 {test_filepath} 未找到。")

    print("\n数据预处理完成！")

if __name__ == '__main__':
    main()