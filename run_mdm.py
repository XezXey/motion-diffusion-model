import numpy as np
import torch as th
import subprocess, os, glob
import argparse

"""
Example: 
python  -m sample.mas 
        --model_path save/nba/nba_diffusion_model/checkpoint_500000.pth 
        --num_samples 10 
        --seed 0 
        --output_dir /data/mint/CR7_data/motion_diffusion/MAS/nba/
"""

def run_mdm(model_path, num_samples, seed, output_dir, input_text):
    subprocess.run([f"python", "-m", f"sample.{args.mode}",
                    "--model_path", model_path,
                    "--num_samples", str(num_samples),
                    "--seed", str(seed),
                    "--output_dir", output_dir,
                    "--input_text", input_text])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--seed', nargs='+', type=int, default=[0])
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--mode', type=str, choices=['generate'], default='generate')
    parser.add_argument('--input_text', type=str, required=True)
    
    args = parser.parse_args()
    
    # use_text_file = False
    if os.path.isdir(args.input_text):
        # use_text_file = True
        text = glob.glob(args.input_text + '/*.txt')
    elif os.path.isfile(args.input_text):
        text = [args.input_text]
    elif isinstance(args.input_text, str):
        text = [args.input_text]
        
    for t in text:
        print(f'Running MDM with seed = {args.seed}')
        for seed in args.seed:
            # if use_text_file:
            #     output_dir = os.path.join(args.output_dir, 
            run_mdm(args.model_path, args.num_samples, seed, input_text=t, output_dir=args.output_dir)
    print('MDM done!')
