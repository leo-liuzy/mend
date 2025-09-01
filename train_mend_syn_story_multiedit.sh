export CUDA_VISIBLE_DEVICES=4

# python -m run +alg=mend +experiment=bio_syn +model=llama3.2-1B-common-date-eos-sft val_steps=100 log_interval=10 val_interval=100 +train_set_size=${train_set_size} mend.shared=False
# llama3.2-1B-common-date-year-after-eos-sft
# llama3.2-1B-common-date-year-after-eos-sft-mid-upper
train_set_size=40_000

n_edits=5
n_locality=1
batch_size=$((n_edits + n_locality))
lr=1e-5

python -m run +alg=mend +experiment=syn_story +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=True train_prefix=4K data.n_edits=${n_edits} batch_size=${batch_size} val_batch_size=${batch_size} lr=${lr}


# python -m run +alg=mend +experiment=syn_story +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-top3 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False train_prefix=4K
