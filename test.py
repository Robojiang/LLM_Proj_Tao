# 先安装：pip install sacrebleu

import sacrebleu

# 5句测试数据：系统预测 vs 唯一参考
hyps = [
    "the cat is on the mat",
    "i like eating apples",
    "he goes to school every day",
    "this is a beautiful day",
    "they are playing football"
]

refs_raw = [
    ["the cat sits on the mat"],      # 第1句参考
    ["i love eating apples"],         # 第2句参考
    ["he goes to school daily"],      # 第3句参考
    ["it is a beautiful day"],        # 第4句参考
    ["they play football"]            # 第5句参考
]


refs_raw_shuttle = [
          # 第1句参考
    ["i love eating apples"],         # 第2句参考
    ["he goes to school daily"], 
    ["the cat sits on the mat"],     # 第3句参考
    ["it is a beautiful day"],        # 第4句参考
    ["they play football"]            # 第5句参考
]

# ❌ 错误用法：直接传 refs_raw
# bleu_wrong = sacrebleu.corpus_bleu(hyps, refs_raw)  # 会算出错误分数

# ✅ 正确用法：必须用 zip(*refs) 转置
refs_stream = list(zip(*refs_raw))  # 结果: [('ref1', 'ref2', ..., 'ref5')]
refs_stream_shuttle = list(zip(*refs_raw_shuttle))

bleu_correct = sacrebleu.corpus_bleu(hyps, refs_raw_shuttle)

print(f"BLEU = {bleu_correct.score:.2f}")
print("\n前3句对比：")
for i in range(3):
    print(f"  Hyp: {hyps[i]}")
    print(f"  Ref: {refs_raw_shuttle[i][0]}")
    print()

# 输出：
# BLEU = 35.96
# 
# 前3句对比：
#   Hyp: the cat is on the mat
#   Ref: the cat sits on the mat
# 
#   Hyp: i like eating apples
#   Ref: i love eating apples
# 
#   Hyp: he goes to school every day
#   Ref: he goes to school daily