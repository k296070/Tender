from utils import *
import os

# If you set EVAL_SAMPLES 0, evaluate over full dataset. (no sampling)
EVAL_SAMPLES = 0
llama_2_dir = os.environ['LLAMA2_PATH']

print('='*10 + ' Llama-2 Tender-INT4 ' + '='*10)
set_symlink_llama('modeling_llama_tender.py')
for SIZE in ['7b', '13b']:
    for SEQLEN in [2048]:
        for DATASET in ["wikitext2", 'ptb']:
            for BITS in [4, 8]:
                DECOMP = llama2_decomp_params(SIZE, BITS)
                cmd = "CUDA_VISIBLE_DEVICES=0 python llama.py "
                cmd += "--model %s/llama-2-%s "%(llama_2_dir, SIZE)
                cmd += "--eval_dataset %s "%(DATASET)
                cmd += "--seq_len %d "%(SEQLEN)
                cmd += "--eval_samples %d "%(EVAL_SAMPLES)
                cmd += "--q_bits %d "%(BITS)
                cmd += "--decomp_factor %d "%(DECOMP)
                cmd += "--chunk_size %d "%(256)
                print(cmd)
                os.system(cmd)
                print("-------------------------------------------")
