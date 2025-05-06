export CUDA_VISIBLE_DEVICES=1

train_set_size=4_000

# python -m run +alg=mend +experiment=bio_syn +model=llama3.2-1B-common-date-eos-sft val_steps=100 log_interval=10 val_interval=100 +train_set_size=${train_set_size} mend.shared=False
# llama3.2-1B-common-date-year-after-eos-sft
# llama3.2-1B-common-date-year-after-eos-sft-mid-upper


python -m run +alg=mend +experiment=syn_story +model=llama3.2-1B-eos-sft-template-format-v3-lr2e-6-sample-10-midupper3 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False
