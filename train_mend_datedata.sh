export CUDA_VISIBLE_DEVICES=2

# python -m run +alg=mend +experiment=musique_injector +model=llama3.2-1B-eos-sft add_icl=True
# python -m run +alg=mend +experiment=musique_dropout_better +model=llama3.2-1B-instruct edit_input=2doc two_doc_at_same_time=True doc_dropout=0 add_icl=False

# python -m run +alg=mend +experiment=musique_dropout +model=llama3.2-1B-eos-sft edit_input=2doc two_doc_at_same_time=True doc_dropout=0.25

# python -m run +alg=mend +experiment=musique_dropout +model=llama3.2-1B-eos-sft edit_input=2doc two_doc_at_same_time=True doc_dropout=1

train_set_size=10_000

# python -m run +alg=mend +experiment=bio_syn +model=llama3.2-1B-common-date-eos-sft val_steps=100 log_interval=10 val_interval=100 +train_set_size=${train_set_size} mend.shared=False
# llama3.2-1B-common-date-year-after-eos-sft
# llama3.2-1B-common-date-year-after-eos-sft-mid-upper

python -m run +alg=mend +experiment=bio_syn_v2 +model=llama3.2-1B-common-date-year-after-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 +train_set_size=${train_set_size} heavy_outerloop=True mend.shared=False 

# mend.shared=False

# python -m run +alg=mend +experiment=musique_propagator_text +model=llama3.2-1B-eos-sft edit_input=hidden outer_loop_add_atomq=True

# python -m run +alg=mend +experiment=zsre +model=llama3.2-1B # +train_size=14326

# python -m run +alg=mend +experiment=musique_propagator_easy +model=llama3.2-1B-eos-sft +outer_loop_add_atomq=True

# python -m run +alg=mend +experiment=musique_propagator_q +model=llama3.2-1B-eos-sft edit_input=first-1hop

# python -m run +alg=mend +experiment=musique_propagator_q +model=llama3.2-1B-eos-sft edit_input=second-1hop