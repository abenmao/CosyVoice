import sys
import os
import torch
from hyperpyyaml import load_hyperpyyaml
import argparse

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'third_party/Matcha-TTS'))

def export_hift(model_dir):
    print(f"Exporting HiFT model from {model_dir}...")
    yaml_path = os.path.join(model_dir, 'cosyvoice3.yaml')
    with open(yaml_path, 'r') as f:
        # We only need hift config
        # But load_hyperpyyaml builds objects.
        # We can use overrides to disable llm/flow to save time/mem
        configs = load_hyperpyyaml(f, overrides={'llm': None, 'flow': None})
    
    hift = configs['hift']
    
    # Load weights
    hift_pt = os.path.join(model_dir, 'hift.pt')
    print(f"Loading weights from {hift_pt}")
    state_dict = torch.load(hift_pt, map_location='cpu')
    print(f"Total keys in hift.pt: {len(state_dict)}")
    if len(state_dict) > 0:
        print(f"Sample keys: {list(state_dict.keys())[:5]}")

    gen_state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items() if k.startswith('generator.')}
    print(f"Keys matching 'generator.': {len(gen_state_dict)}")
    
    if len(gen_state_dict) == 0:
        # Maybe keys usually don't have 'generator.'?
        print("Warning: No keys starting with 'generator.' found. Trying raw keys.")
        gen_state_dict = state_dict

    try:
        hift.load_state_dict(gen_state_dict, strict=True)
    except Exception as e:
        print(f"Strict load failed: {e}")
        # Try strict=False to see what matches
        hilt_keys = set(hift.state_dict().keys())
        loaded_keys = set(gen_state_dict.keys())
        print(f"Model keys: {len(hilt_keys)}")
        print(f"Loaded keys: {len(loaded_keys)}")
        print(f"Intersection: {len(hilt_keys.intersection(loaded_keys))}")
        raise e
    hift.eval()
    
    # Export Final
    final_onnx = os.path.join(model_dir, 'hift.generator.fp32.final.onnx')
    print(f"Exporting to {final_onnx}...")
    hift.export_onnx(final_onnx, finalize=True)
    
    # Export Stream
    stream_onnx = os.path.join(model_dir, 'hift.generator.fp32.stream.onnx')
    print(f"Exporting to {stream_onnx}...")
    hift.export_onnx(stream_onnx, finalize=False)
    print("Export done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    args = parser.parse_args()
    
    export_hift(args.model_dir)
