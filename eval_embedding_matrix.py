import os
import pickle
import numpy as np

# ========== 1. 路径与参数（按你实际目录改） ==========
PROCESSED_DIR = './processed_data'          # 与 build 脚本保持一致
LANG = 'zh'                                 # 'en' 或 'zh'
TEST_WORD = '走'                          # 想查的词
TOPK = 10                                   # 关联度最高的 N 个词

# 如果下面文件不存在会抛 FileNotFoundError，方便一眼看出问题
VOCAB_PATH = os.path.join(PROCESSED_DIR, f'{LANG}_vocab.pkl')
MATRIX_PATH = os.path.join(PROCESSED_DIR, f'{LANG}_embedding_matrix.npy')

# ========== 2. 工具函数 ==========
def l2_normalize(matrix):
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1
    return matrix / norm

def load_vocab_and_matrix():
    with open(VOCAB_PATH, 'rb') as f:
        word2id = pickle.load(f)
    id2word = {i: w for w, i in word2id.items()}
    matrix = np.load(MATRIX_PATH)
    matrix = l2_normalize(matrix.astype(np.float32))
    return word2id, id2word, matrix

def most_similar(word, topk=TOPK):
    word2id, id2word, matrix = load_vocab_and_matrix()
    if word not in word2id:
        print(f'"{word}" 不在词汇表里')
        return
    idx = word2id[word]
    vec = matrix[idx]
    sims = matrix @ vec          # 余弦相似度（已归一化）
    best = np.argsort(-sims)[:topk+1]
    print(f'\n与 "{word}" 最接近的 {topk} 个词：')
    cnt = 0
    for ix in best:
        if ix == idx:
            continue
        print(f'{cnt+1:2d}.  {id2word[ix]:<12}  {sims[ix]:.4f}')
        cnt += 1
        if cnt >= topk:
            break

# ========== 3. 跑测试 ==========
if __name__ == '__main__':
    most_similar(TEST_WORD)