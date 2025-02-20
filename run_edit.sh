export CUDA_VISIBLE_DEVICES=3

n_val=1000
task=musique
# task=zsre
# archive=2025-02-10_08-19-14_2641409766
archive=2025-02-11_23-27-06_6306217737

exp_dir_name=llama3.2-1B_on_musiqueQonly #
# prompt=no
prompt=no

# sft(q_p, a_p)
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True

# clm(q_p + a_p)
python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=question generation.prompt=${prompt} +do_generation=True

# 
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt}

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=question generation.prompt=${prompt}

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=2doc generation.prompt=${prompt}

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=1doc generation.prompt=${prompt}

# exp_dir_name=llama3.2-1B_on_musique_p0.25
# archive=2025-02-12_21-32-59_3112383942


# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=2doc generation.prompt=${prompt}

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=1doc generation.prompt=${prompt}


# exp_dir_name=llama3.2-1B_on_musique_p0.5
# archive=2025-02-12_20-39-47_2326231633

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=2doc generation.prompt=${prompt}

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=1doc generation.prompt=${prompt}
