export CUDA_VISIBLE_DEVICES=1


train_set_size=10000


# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True # mend.shared=False 


# python -m run +alg=mend +experiment=ripple_edits +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False 


python -m run +alg=mend +experiment=ripple_edits_mend +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=False 
