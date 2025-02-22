export CUDA_VISIBLE_DEVICES=1

python -m run +alg=mend +experiment=musique_dropout_better +model=llama3.2-1B-instruct edit_input=2doc two_doc_at_same_time=True doc_dropout=0 add_icl=False


python -m run +alg=mend +experiment=zsre +model=llama3.2-1B # +train_size=14326