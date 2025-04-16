export CUDA_VISIBLE_DEVICES=3


train_set_size=10000


python -m run +alg=mend +experiment=ripple_edits_w_aug +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid1 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False
# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid2 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid-lower val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False 

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-bottom val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False 

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.rank=3840

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.n_hidden=2

# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.n_hidden=4

# python -m run +alg=mend +experiment=ripple_edits_mend +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=False 
