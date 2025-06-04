#!/bin/bash
#SBATCH -J qwen-max       # Job name
#SBATCH -o slurm-outputs/%x.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/%x.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1 
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH -A CCR25005       # Allocation name (req'd if you have more than 1)


export CUDA_VISIBLE_DEVICES=0

train_set_size=40_000

# python -m run +alg=mend +experiment=syn_story +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-max val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K

# midupper
# python -m run +alg=mend +experiment=syn_story +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K

# top
python -m run +alg=mend +experiment=syn_story +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-top val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False seed=1

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False mend.rank=480

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False mend.rank=240

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False mend.rank=120
# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid1 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False
# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid2 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid-lower val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False 

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-bottom val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False 

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.rank=3840

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.n_hidden=2

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.n_hidden=4

# python -m run +alg=mend +experiment=ripple_edits_all_mend +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=False 
