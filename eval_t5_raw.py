import os, json, torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from tqdm import tqdm
import sacrebleu
import argparse

# 尝试导入 PEFT
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

def main():
    parser = argparse.ArgumentParser()
    # --- 修改：参数名更通用 ---
    parser.add_argument('--model_path', default='./runs/20251218_121749_mt5-finetune-raw_google-mt5-small_lora_raw/best_model', help="模型文件夹路径 或 .pt 检查点文件路径")
    # --- 新增：当评估 .pt 文件时，需要指定基础模型 ---
    parser.add_argument('--base_model_name', default='google/mt5-small', help="基础模型名称")
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--test_file', default='trian_test.jsonl', help="测试文件名")
    parser.add_argument('--max_len', type=int, default=80, help="生成最大长度")
    parser.add_argument('--num_beams', type=int, default=4, help="Beam search 的 beam 数量")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cache_dir', default='./hf_cache', help="HuggingFace 缓存目录")
    args = parser.parse_args()

    # --- 核心修改：智能加载模型 ---
    if os.path.isdir(args.model_path):
        print(f"加载已保存的模型文件夹: {args.model_path}")
        
        # 检查是否是 LoRA 模型 (存在 adapter_config.json)
        is_lora = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
        
        if is_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("检测到 LoRA 模型，但未安装 peft 库。")
            
            print("🚀 检测到 LoRA 适配器，正在加载基础模型和适配器...")
            
            # 1. 加载基础模型
            # 尝试从 adapter_config.json 获取基础模型名称，如果失败则使用 args.base_model_name
            try:
                config = PeftConfig.from_pretrained(args.model_path)
                base_model_path = config.base_model_name_or_path
            except Exception:
                base_model_path = args.base_model_name
            
            print(f"基础模型: {base_model_path} (Cache: {args.cache_dir})")
            
            # --- 核心修复：手动寻找缓存的快照路径 ---
            # 如果 base_model_path 是模型名（如 google/mt5-small），尝试找到其在 cache 中的真实路径
            if not os.path.exists(base_model_path):
                model_cache_name = "models--" + base_model_path.replace("/", "--")
                snapshot_dir = os.path.join(args.cache_dir, model_cache_name, "snapshots")
                
                if os.path.exists(snapshot_dir):
                    snapshots = os.listdir(snapshot_dir)
                    
                    # 寻找包含配置文件的快照
                    config_snap = None
                    for snap in snapshots:
                        if os.path.exists(os.path.join(snapshot_dir, snap, "config.json")):
                            config_snap = os.path.join(snapshot_dir, snap)
                            break
                    
                    # 寻找包含权重的快照
                    weight_snap = None
                    weight_file = None
                    for snap in snapshots:
                        s_path = os.path.join(snapshot_dir, snap)
                        if os.path.exists(os.path.join(s_path, "model.safetensors")):
                            weight_snap = s_path
                            weight_file = "model.safetensors"
                            break
                        elif os.path.exists(os.path.join(s_path, "pytorch_model.bin")):
                            weight_snap = s_path
                            weight_file = "pytorch_model.bin"
                            break
                    
                    print(f"🔍 缓存分析: Config在 {os.path.basename(config_snap) if config_snap else 'None'}, Weights在 {os.path.basename(weight_snap) if weight_snap else 'None'}")

                    if config_snap and weight_snap:
                        if config_snap == weight_snap:
                            # 完美情况：都在同一个目录
                            base_model_path = config_snap
                            tokenizer = MT5Tokenizer.from_pretrained(base_model_path, local_files_only=True)
                            model = MT5ForConditionalGeneration.from_pretrained(base_model_path, local_files_only=True)
                        else:
                            # 分裂情况：手动拼接
                            print("⚠️ 检测到缓存分裂（Config和Weights在不同目录），正在手动组装...")
                            from transformers import AutoConfig
                            from safetensors.torch import load_file
                            
                            tokenizer = MT5Tokenizer.from_pretrained(config_snap, local_files_only=True)
                            config = AutoConfig.from_pretrained(config_snap, local_files_only=True)
                            model = MT5ForConditionalGeneration(config)
                            
                            weight_path = os.path.join(weight_snap, weight_file)
                            if weight_file.endswith(".safetensors"):
                                state_dict = load_file(weight_path)
                            else:
                                state_dict = torch.load(weight_path, map_location="cpu")
                            
                            model.load_state_dict(state_dict, strict=False)
                    else:
                        # 无法修复，回退到默认逻辑（可能会报错）
                        print("❌ 无法在缓存中找到完整的模型文件")
                        if config_snap: base_model_path = config_snap # 至少尝试加载config
            
            if 'model' not in locals(): # 如果上面没有成功加载 model
                tokenizer = MT5Tokenizer.from_pretrained(base_model_path, local_files_only=True)
                model = MT5ForConditionalGeneration.from_pretrained(base_model_path, local_files_only=True)
            
            # 2. 加载 LoRA 适配器
            model = PeftModel.from_pretrained(model, args.model_path, local_files_only=True)
            
        else:
            # 全量微调模型
            print("🚀 加载全量微调模型...")
            tokenizer = MT5Tokenizer.from_pretrained(args.model_path, local_files_only=True)
            model = MT5ForConditionalGeneration.from_pretrained(args.model_path, use_safetensors=True, local_files_only=True)
            
        model.to(args.device)

    elif args.model_path.endswith('.pt'):
        # 情况 2: 输入是 .pt 检查点文件
        print(f"加载检查点文件: {args.model_path}")
        print(f"使用基础模型 '{args.base_model_name}' 构建结构 (Cache: {args.cache_dir})")
        
        # 先加载基础模型结构和分词器
        tokenizer = MT5Tokenizer.from_pretrained(args.base_model_name, cache_dir=args.cache_dir, local_files_only=True)
        model = MT5ForConditionalGeneration.from_pretrained(args.base_model_name, cache_dir=args.cache_dir, local_files_only=True)
        
        # 加载检查点中的权重
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        # 处理可能的 key 不匹配 (例如带有 'module.' 前缀)
        state_dict = checkpoint['model_state_dict']
        # 如果是 LoRA 的 checkpoint，这里可能需要特殊处理，但通常 .pt 是全量保存或者只保存了 adapter
        # 假设这里是全量或者用户知道自己在做什么
        
        model.load_state_dict(state_dict, strict=False)
        model.to(args.device)
    else:
        raise ValueError("无效的 --model_path，必须是文件夹或 .pt 文件")

    model.eval()

    hyps, refs, sources = [], [], []
    test_path = os.path.join(args.data_dir, args.test_file)
    print(f"加载测试数据: {test_path}")
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="翻译中"):
            pair = json.loads(line)
            source_text = pair['zh']
            ref_text = pair['en']
            
            prompted_text = "translate Chinese to English: " + source_text

            input_ids = tokenizer(prompted_text, return_tensors='pt', max_length=args.max_len, truncation=True).input_ids.to(args.device)
            
            outputs = model.generate(
                input_ids=input_ids, 
                max_length=args.max_len,
                num_beams=args.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.0
            )
            hyp = tokenizer.decode(outputs[0], skip_special_tokens=True)

            hyps.append(hyp)
            sources.append(source_text)
            refs.append([ref_text])

    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    print(f"\n{'='*50}")
    print(f"T5 BLEU Score: {bleu.score:.2f}")
    print(f"{'='*50}\n")
    
    print("示例翻译（前 5 条）:")
    for i in range(min(5, len(hyps))):
        print(f"\n--- 样本 {i+1} ---")
        print(f"源: {sources[i]}")
        print(f"预测: {hyps[i]}")
        print(f"参考: {refs[i][0]}")

if __name__ == '__main__':
    main()