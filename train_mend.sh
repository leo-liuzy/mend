export CUDA_VISIBLE_DEVICES=2

# python -m run +alg=mend +experiment=musique_dropout +model=llama3.2-1B edit_input=question two_doc_at_same_time=False


python -m run +alg=mend +experiment=zsre +model=llama3.2-1B # +train_size=14326