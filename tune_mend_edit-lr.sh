export CUDA_VISIBLE_DEVICES=5

# python -m run +alg=mend +experiment=musique_dropout +model=llama3.2-1B edit_input=2doc two_doc_at_same_time=True doc_dropout=0.25

for edit_lr in 1e-5 1e-7 1e-4 1e-8
do

python -m run +alg=mend +experiment=musique +model=llama3.2-1B edit_input=2doc two_doc_at_same_time=True edit_lr=${edit_lr}

done