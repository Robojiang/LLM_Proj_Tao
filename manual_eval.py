import os
import subprocess
import re
import csv
import argparse
from tqdm import tqdm

# ==================================================================================
# 在此处手动配置所有需要评估的实验
# 格式:
# {
#     "name": "显示在报告中的名称",
#     "ckpt": "checkpoint文件的相对或绝对路径",
#     "decode": "greedy" 或 "beam",
#     "beam_size": 整数 (仅在 decode='beam' 时生效),
#     "device": "cuda" 或 "cpu" (可选，默认 cuda)
# }
# ==================================================================================

EXPERIMENTS = [
    # --- RNN Experiments ---
    # {
    #     "name": "RNN Dot Attention (TF=0.5)",
    #     "ckpt": "runs/20251224_225052_rnn_dot_0.5_256/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # {
    #     "name": "RNN General Attention (TF=0.5)",
    #     "ckpt": "runs/20251224_233107_rnn_general_0.5_256/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # {
    #     "name": "RNN Additive Attention (TF=0.5)",
    #     "ckpt": "runs/20251225_001421_rnn_additive_0.5_256/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # # --- RNN Teacher Forcing Ablation ---
    # {
    #     "name": "RNN Additive Attention (TF=1.0)",
    #     "ckpt": "runs/20251225_010120_rnn_additive_1_256/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # {
    #     "name": "RNN Additive Attention (TF=0.0)",
    #     "ckpt": "runs/20251225_014811_rnn_additive_0_256/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },

    # # --- RNN decode Ablation ---
    # {
    #     "name": "RNN Additive Attention (TF=0.5)",
    #     "ckpt": "runs/20251225_001421_rnn_additive_0.5_256/best.pt",
    #     "decode": "beam", "beam_size": 10
    # },
    # {
    #     "name": "RNN Additive Attention (TF=0.5)",
    #     "ckpt": "runs/20251225_001421_rnn_additive_0.5_256/best.pt",
    #     "decode": "greedy", "beam_size": 1
    # },

    # # ============================================================
    # # Transformer Experiments
    # # ============================================================

    # # --- Group 1: Architecture Ablation ---
    # {
    #     "name": "Transformer (Abs Pos + LayerNorm)",
    #     "ckpt": "runs/20251224_232445_transformer_absolute_layernorm_128_0.0001_512/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # {
    #     "name": "Transformer (Rel Pos + LayerNorm)",
    #     "ckpt": "runs/20251225_011244_transformer_relative_layernorm_128_0.0001_512/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # {
    #     "name": "Transformer (Rel Pos + RMSNorm) [Baseline]",
    #     "ckpt": "runs/20251225_025324_transformer_relative_rmsnorm_128_0.0001_512/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },

    # # --- Group 2: Batch Size Sensitivity ---
    # {
    #     "name": "Transformer (Batch=64)",
    #     "ckpt": "runs/20251225_043357_transformer_relative_rmsnorm_64_0.0001_512/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # {
    #     "name": "Transformer (Batch=256)",
    #     "ckpt": "runs/20251225_071447_transformer_relative_rmsnorm_256_0.0001_512/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },

    # # --- Group 3: Learning Rate Sensitivity ---
    # {
    #     "name": "Transformer (LR=5e-5)",
    #     "ckpt": "runs/20251225_083940_transformer_relative_rmsnorm_128_5e-05_512/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },
    # {
    #     "name": "Transformer (LR=1.5e-4)",
    #     "ckpt": "runs/20251225_101629_transformer_relative_rmsnorm_128_0.00015_512/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },

    # # --- Group 4: Model Scales ---
    # {
    #     "name": "Transformer (Tiny: d=256, L=3)",
    #     "ckpt": "runs/20251225_115215_transformer_relative_rmsnorm_128_0.0001_256/best.pt",
    #     "decode": "beam", "beam_size": 5
    # },

    {
        "name": "Transformer (Tiny: d=768, L=9)",
        "ckpt": "./runs/20251225_124943_transformer_relative_rmsnorm_128_0.0001_768/best.pt",
        "decode": "beam", "beam_size": 5
    },
]

def get_bleu_score(output):
    # 匹配 eval.py 输出中的 "BLEU = 25.43" 格式
    match = re.search(r"BLEU = (\d+\.\d+)", output)
    if match:
        return float(match.group(1))
    return None

def check_ckpt_exists(ckpt_path):
    """检查 checkpoint 是否存在，如果不存在尝试寻找 epoch_*.pt"""
    if os.path.exists(ckpt_path):
        return ckpt_path
    
    # 如果指定的文件不存在，尝试找同目录下的其他 pt 文件
    dir_path = os.path.dirname(ckpt_path)
    if not os.path.exists(dir_path):
        return None
        
    print(f"Warning: {ckpt_path} not found. Searching for alternatives in {dir_path}...")
    pts = [f for f in os.listdir(dir_path) if f.endswith('.pt') and 'epoch' in f]
    if pts:
        pts.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        alt_path = os.path.join(dir_path, pts[-1])
        print(f"  -> Found alternative: {alt_path}")
        return alt_path
    
    if os.path.exists(os.path.join(dir_path, "last.pt")):
        alt_path = os.path.join(dir_path, "last.pt")
        print(f"  -> Found alternative: {alt_path}")
        return alt_path
        
    return None

def main():
    parser = argparse.ArgumentParser(description="手动配置评估脚本")
    parser.add_argument("--output_file", default="manual_evaluation_report.csv", help="结果保存文件")
    parser.add_argument("--default_device", default="cuda", help="默认设备")
    args = parser.parse_args()

    results = []
    
    print(f"准备评估 {len(EXPERIMENTS)} 个实验配置...")
    
    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Evaluating: {exp['name']}")
        
        ckpt_path = check_ckpt_exists(exp['ckpt'])
        if not ckpt_path:
            print(f"  [Error] Checkpoint not found: {exp['ckpt']}")
            results.append({
                "name": exp['name'],
                "bleu": -1.0,
                "details": "Checkpoint not found"
            })
            continue
            
        device = exp.get('device', args.default_device)
        decode_method = exp.get('decode', 'greedy')
        beam_size = exp.get('beam_size', 4)
        
        # 自动在名称中添加解码信息，防止重名
        display_name = exp['name']
        if decode_method == 'greedy':
            display_name += " [Greedy]"
        else:
            display_name += f" [Beam={beam_size}]"
        
        cmd = [
            "python", "eval.py",
            "--ckpt", ckpt_path,
            "--device", device,
            "--decode", decode_method,
            "--beam_size", str(beam_size)
        ]
        
        print(f"  Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  [Error] Evaluation failed")
                print(result.stderr)
                results.append({
                    "name": display_name,
                    "bleu": -1.0,
                    "details": "Runtime Error"
                })
                continue

            bleu = get_bleu_score(result.stdout)
            
            if bleu is not None:
                print(f"  -> BLEU: {bleu}")
                results.append({
                    "name": display_name,
                    "bleu": bleu,
                    "details": f"{decode_method} (b={beam_size})" if decode_method == 'beam' else "greedy"
                })
            else:
                print(f"  [Warning] Could not parse BLEU score")
                results.append({
                    "name": display_name,
                    "bleu": 0.0,
                    "details": "Parse Error"
                })
                
        except Exception as e:
            print(f"  [Exception] {e}")
            
    # 保存结果
    if results:
        # 按 BLEU 降序
        results.sort(key=lambda x: x['bleu'], reverse=True)
        
        with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["name", "bleu", "details"])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n✅ 评估完成！结果已保存至 {args.output_file}")
        print("\n🏆 排行榜:")
        print(f"{'Rank':<5} {'BLEU':<8} {'Name'}")
        print("-" * 50)
        for i, res in enumerate(results):
            if res['bleu'] >= 0:
                print(f"{i+1:<5} {res['bleu']:<8.2f} {res['name']} ({res['details']})")
            else:
                print(f"{i+1:<5} {'ERR':<8} {res['name']}")

if __name__ == "__main__":
    main()
