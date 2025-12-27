import os
import json
import torch
import argparse
import sys
from tqdm import tqdm
import sacrebleu

# Add current directory to sys.path
sys.path.append(os.getcwd())

# Import from existing modules
from eval import load_model as load_rnn_tra_model
from eval import ids_to_tokens, detokenize, PAD, UNK, SOS, EOS, collate_fn, NMTDataset
from torch.utils.data import DataLoader

# --- MT5 Loading Logic (Adapted from eval_t5_raw.py) ---
def load_mt5_model_smart(model_path, base_model_name, cache_dir, device):
    from transformers import MT5Tokenizer, MT5ForConditionalGeneration
    try:
        from peft import PeftModel, PeftConfig
        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False

    print(f"Loading MT5 from: {model_path}")
    
    # Check for LoRA
    is_lora = os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA model detected but 'peft' is not installed.")
        
        print("🚀 LoRA adapter detected. Loading base model...")
        
        # Try to get base model path from config
        try:
            config = PeftConfig.from_pretrained(model_path)
            base_model_path = config.base_model_name_or_path
        except Exception:
            base_model_path = base_model_name
            
        print(f"Base model target: {base_model_path}")

        # Smart Cache Search (to avoid internet)
        if not os.path.exists(base_model_path):
            model_cache_name = "models--" + base_model_path.replace("/", "--")
            snapshot_dir = os.path.join(cache_dir, model_cache_name, "snapshots")
            
            config_snap = None
            weight_snap = None
            weight_file = None

            if os.path.exists(snapshot_dir):
                snapshots = os.listdir(snapshot_dir)
                # Find config
                for snap in snapshots:
                    if os.path.exists(os.path.join(snapshot_dir, snap, "config.json")):
                        config_snap = os.path.join(snapshot_dir, snap)
                        break
                # Find weights
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
            
            if config_snap and weight_snap:
                print(f"✅ Found base model in cache: {config_snap}")
                if config_snap == weight_snap:
                    base_model_path = config_snap
                    tokenizer = MT5Tokenizer.from_pretrained(base_model_path, local_files_only=True)
                    model = MT5ForConditionalGeneration.from_pretrained(base_model_path, local_files_only=True)
                else:
                    print("⚠️ Cache split detected. Assembling manually...")
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
                print("❌ Base model not found in local cache. Fallback to online/config path.")
                # Fallback: try loading tokenizer from adapter path if available (user case)
                if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
                     print("Using tokenizer from adapter folder.")
                     tokenizer = MT5Tokenizer.from_pretrained(model_path, local_files_only=True)
                     # We still need the base model weights though...
                     # If user really has no base model, this will fail or try to download.
                     model = MT5ForConditionalGeneration.from_pretrained(base_model_path) # Might download
                else:
                     tokenizer = MT5Tokenizer.from_pretrained(base_model_path)
                     model = MT5ForConditionalGeneration.from_pretrained(base_model_path)

        else:
            # Base model path exists locally
            tokenizer = MT5Tokenizer.from_pretrained(base_model_path, local_files_only=True)
            model = MT5ForConditionalGeneration.from_pretrained(base_model_path, local_files_only=True)

        # Load Adapter
        model = PeftModel.from_pretrained(model, model_path, local_files_only=True)

    else:
        # Full model
        print("Loading full model...")
        tokenizer = MT5Tokenizer.from_pretrained(model_path, local_files_only=True)
        model = MT5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)

    model.to(device)
    model.eval()
    return model, tokenizer

# --- Evaluation Functions ---

def evaluate_rnn_transformer(model_type, model_path, data_path, args):
    print(f"\nEvaluating {model_type.upper()}...")
    if not os.path.exists(model_path):
        print(f"Skipping {model_type}: Path {model_path} not found.")
        return

    # Load Checkpoint
    ckpt = torch.load(model_path, map_location=args.device)
    cfg = ckpt['config']
    zh_vocab = ckpt['zh_vocab']
    en_vocab = ckpt['en_vocab']
    id2en = {v: k for k, v in en_vocab.items()}
    pad_src, pad_tgt = zh_vocab[PAD], en_vocab[PAD]

    # Load Data
    test_ds = NMTDataset(data_path, zh_vocab, args.max_len)
    # Force batch_size=1 for RNN/Transformer because their greedy_decode/beam_search 
    # implementations might not support batching (or support it partially).
    # Given test set is small (~200 lines), this is fine.
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_src))

    # Load Model
    model = load_rnn_tra_model(cfg, zh_vocab, en_vocab, pad_src, pad_tgt, args.device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    hyps, refs = [], []
    
    with torch.no_grad():
        for src, src_lens, tgt_ref in tqdm(loader, desc=f"Testing {model_type}"):
            src, src_lens = src.to(args.device), src_lens.to(args.device)
            
            if args.decode == 'greedy':
                out_ids = model.greedy_decode(src, src_lens, args.max_len, en_vocab[SOS], en_vocab[EOS])
            else:
                # Beam search usually returns [batch, beam, len] or similar, need to handle batch
                # The provided beam_search implementation in eval.py might be single-sample or batch.
                # Assuming batch support or iterating. 
                # If model.beam_search supports batch:
                out_ids = model.beam_search(src, src_lens, args.max_len, en_vocab[SOS], en_vocab[EOS], beam_size=args.beam_size)
            
            # Decode
            for i in range(len(out_ids)):
                toks = ids_to_tokens(out_ids[i].tolist(), id2en)
                if toks and toks[0] == SOS: toks = toks[1:]
                hyps.append(detokenize(toks))
                refs.append([detokenize(tgt_ref[i])])

    # BLEU
    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    print(f"[{model_type.upper()}] BLEU: {bleu.score:.2f}")
    print("Examples:")
    for i in range(min(5, len(hyps))):
        print(f"  Ref: {refs[i][0]}")
        print(f"  Hyp: {hyps[i]}")
        print("-" * 20)


def evaluate_mt5(model_path, data_path, args):
    print(f"\nEvaluating MT5...")
    if not os.path.exists(model_path):
        print(f"Skipping MT5: Path {model_path} not found.")
        return

    # Load Model
    try:
        model, tokenizer = load_mt5_model_smart(model_path, args.base_model_name, args.cache_dir, args.device)
    except Exception as e:
        print(f"Failed to load MT5: {e}")
        return

    hyps, refs = [], []
    
    # Read Data (Raw JSONL)
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Batch processing
    batch_size = args.batch_size
    for i in tqdm(range(0, len(lines), batch_size), desc="Testing MT5"):
        batch_lines = lines[i:i+batch_size]
        inputs = []
        batch_refs = []
        
        for line in batch_lines:
            pair = json.loads(line)
            inputs.append("translate Chinese to English: " + pair['zh'])
            batch_refs.append(pair['en'])

        # Tokenize
        input_ids = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=args.max_len).input_ids.to(args.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=args.max_len,
                num_beams=args.beam_size if args.decode == 'beam' else 1,
                early_stopping=True
            )
        
        # Decode
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        hyps.extend(decoded)
        for r in batch_refs:
            refs.append([r])

    # BLEU
    bleu = sacrebleu.corpus_bleu(hyps, list(zip(*refs)))
    print(f"[MT5] BLEU: {bleu.score:.2f}")
    print("Examples:")
    for i in range(min(5, len(hyps))):
        print(f"  Ref: {refs[i][0]}")
        print(f"  Hyp: {hyps[i]}")
        print("-" * 20)


def main():
    parser = argparse.ArgumentParser(description="Evaluation script for NMT models")
    
    # Common Args
    parser.add_argument('--model_type', type=str, default='all', choices=['rnn', 'transformer', 'mt5', 'all'])
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    parser.add_argument('--rnn_path', default='best_weight/RNN/best.pt')
    parser.add_argument('--transformer_path', default='best_weight/Transformer/best.pt')
    parser.add_argument('--mt5_path', default='best_weight/MT5/best_model')
    
    # Data Paths
    parser.add_argument('--processed_data', default='processed_data/test.jsonl', help="For RNN/Transformer")
    parser.add_argument('--raw_data', default='data/test.jsonl', help="For MT5")
    
    # MT5 Specific
    parser.add_argument('--base_model_name', default='google/mt5-small')
    parser.add_argument('--cache_dir', default='./hf_cache')

    args = parser.parse_args()

    # Run Evaluations
    if args.model_type in ['rnn', 'all']:
        evaluate_rnn_transformer('rnn', args.rnn_path, args.processed_data, args)
    
    if args.model_type in ['transformer', 'all']:
        evaluate_rnn_transformer('transformer', args.transformer_path, args.processed_data, args)
        
    if args.model_type in ['mt5', 'all']:
        evaluate_mt5(args.mt5_path, args.raw_data, args)

if __name__ == "__main__":
    main()
