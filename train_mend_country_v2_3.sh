export CUDA_VISIBLE_DEVICES=1

train_set_size=3_000

# python -m run +alg=mend +experiment=bio_syn +model=llama3.2-1B-common-date-eos-sft val_steps=100 log_interval=10 val_interval=100 +train_set_size=${train_set_size} mend.shared=False
# llama3.2-1B-common-date-year-after-eos-sft
# llama3.2-1B-common-date-year-after-eos-sft-mid-upper


python -m run +alg=mend +experiment=country_syn_v2 +model=llama3.2-1B-eos-sft-country-template-format-lr2e-6-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False n_template=7 n_seen_pair=100

for n_template in 1
do
for n_seen_pair in 200 100
do

python -m run +alg=mend +experiment=country_syn_v2 +model=llama3.2-1B-eos-sft-country-template-format-lr2e-6-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False n_template=${n_template} n_seen_pair=${n_seen_pair}

done
done

# python -m run +alg=mend +experiment=country_syn +model=llama3.2-1B-common-country-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=False mend.shared=False