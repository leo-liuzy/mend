export CUDA_VISIBLE_DEVICES=1

declare -A name2id=(
    [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    [llama3.2-1B_on_zsre-14K]=2025-02-17_16-06-54_298004526
    [llama3.2-1B_on_musiqueQonly]=2025-02-11_23-27-06_6306217737

    # edit_lr
    [llama3.2-1B_on_musique_editlr1e-4]=2025-02-18_00-48-02_2245122483
    [llama3.2-1B_on_musique_editlr1e-5]=2025-02-17_15-56-35_0163092923
    [llama3.2-1B_on_musique_editlr1e-6]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musique_editlr1e-7]=2025-02-18_06-35-03_2633255081
    [llama3.2-1B_on_musique_editlr1e-8]=2025-02-18_15-01-06_0891593193
    # lr_lr
    [llama3.2-1B_on_musique_lrlr1e-2]=2025-02-18_10-54-20_5105925359
    [llama3.2-1B_on_musique_lrlr1e-3]=2025-02-17_16-11-33_8776393723
    [llama3.2-1B_on_musique_lrlr1e-4]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musique_lrlr1e-5]=2025-02-18_01-30-25_7593913958
    [llama3.2-1B_on_musique_lrlr1e-6]=2025-02-18_00-47-40_0516168634
)

n_val=1000
task=musique
# task=zsre
# archive=2025-02-10_08-19-14_2641409766
exp_dir_name="llama3.2-1B_on_zsre-14K"
archive=${name2id[$exp_dir_name]}
# prompt=no
prompt=no

# sft(q_p, a_p)
python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=False

# clm(q_p + a_p)
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=question generation.prompt=${prompt} +do_generation=True

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
