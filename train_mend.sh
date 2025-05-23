export CUDA_VISIBLE_DEVICES=0

# python -m run +alg=mend +experiment=musique_injector +model=llama3.2-1B-eos-sft add_icl=True
# python -m run +alg=mend +experiment=musique_dropout_better +model=llama3.2-1B-instruct edit_input=2doc two_doc_at_same_time=True doc_dropout=0 add_icl=False

# python -m run +alg=mend +experiment=musique_dropout +model=llama3.2-1B-eos-sft edit_input=2doc two_doc_at_same_time=True doc_dropout=0.25

# python -m run +alg=mend +experiment=musique_dropout +model=llama3.2-1B-eos-sft edit_input=2doc two_doc_at_same_time=True doc_dropout=1

# python -m run +alg=mend +experiment=musique_combiner_text +model=llama3.2-1B-eos-sft edit_input=all outer_loop_add_atomq=True

# python -m run +alg=mend +experiment=musique_propagator_text +model=llama3.2-1B-eos-sft edit_input=hidden outer_loop_add_atomq=True

# python -m run +alg=mend +experiment=zsre +model=llama3.2-1B-eos-sft-mid-upper val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=False data.rephrase=True 

python -m run +alg=mend +experiment=zsre +model=llama3.2-1B-eos-sft val_steps=100 log_interval=10 val_interval=100 early_stop_patience=2000 mend.shared=True data.rephrase=True

# python -m run +alg=mend +experiment=musique_propagator_easy +model=llama3.2-1B-eos-sft +outer_loop_add_atomq=True

# python -m run +alg=mend +experiment=musique_propagator_q +model=llama3.2-1B-eos-sft edit_input=first-1hop

# python -m run +alg=mend +experiment=musique_propagator_q +model=llama3.2-1B-eos-sft edit_input=second-1hop