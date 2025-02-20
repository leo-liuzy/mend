export CUDA_VISIBLE_DEVICES=3

# python -m run +alg=mend +experiment=musique_dropout +model=llama3.2-1B edit_input=2doc two_doc_at_same_time=True doc_dropout=0.25

for lr_lr in 1e-2 # 1e-3 1e-5
do
python -m run +alg=mend +experiment=musique +model=llama3.2-1B edit_input=2doc two_doc_at_same_time=True lr_lr=$lr_lr

done
