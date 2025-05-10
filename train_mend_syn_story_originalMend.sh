export CUDA_VISIBLE_DEVICES=3

train_set_size=40_000

# python -m run +alg=mend +experiment=bio_syn +model=llama3.2-1B-common-date-eos-sft val_steps=100 log_interval=10 val_interval=100 +train_set_size=${train_set_size} mend.shared=False
# llama3.2-1B-common-date-year-after-eos-sft
# llama3.2-1B-common-date-year-after-eos-sft-mid-upper


# python -m run +alg=mend +experiment=syn_story_mend +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-midupper3 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} mend.shared=False train_prefix=4K

python -m run +alg=mend +experiment=zsre +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-top3 val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=True data.rephrase=True