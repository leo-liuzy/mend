export CUDA_VISIBLE_DEVICES=3

n_val=1000
task=musique
# task=zsre
# urial
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=2025-02-10_08-19-14_2641409766 eval_only=True generation.save_dir=exp_output/llama3.2-1B/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=urial

python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=2025-02-10_08-19-14_2641409766 eval_only=True generation.save_dir=exp_output/llama3.2-1B/${task} val_steps=${n_val} edit_loss=clm edit_input=question generation.prompt=urial

python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=2025-02-10_08-19-14_2641409766 eval_only=True generation.save_dir=exp_output/llama3.2-1B/${task} val_steps=${n_val} edit_loss=clm edit_input=2doc generation.prompt=urial

python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=2025-02-10_08-19-14_2641409766 eval_only=True generation.save_dir=exp_output/llama3.2-1B/${task} val_steps=${n_val} edit_loss=clm edit_input=1doc generation.prompt=urial
