export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    [llama3.2-1B_on_zsre-14K]=2025-02-17_16-06-54_298004526
    [llama3.2-1B_on_musiqueQonly]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musiqueQ_w-eos]=2025-02-20_21-05-21_5456934010
    [musique_propagator_q]=2025-02-25_11-23-08_2966052990
    [musique_injector]=2025-02-25_02-02-08_2733371512
    [eos-sft_musique_propagator_p0]=2025-02-24_07-26-33_0114923306
    # metatrain with 2 doc
    [musique_propagator_p0]=2025-02-22_06-58-48_2302322883
    # [musique_propagator_p0_w-newline]=2025-02-22_21-40-54_1897999909
    [musique_propagator_p0_w-newline_icl]=2025-02-22_21-40-16_9068188311
    # edit_lr no EOS
    [llama3.2-1B_on_musique_editlr1e-4]=2025-02-18_00-48-02_2245122483
    [llama3.2-1B_on_musique_editlr1e-5]=2025-02-17_15-56-35_0163092923
    [llama3.2-1B_on_musique_editlr1e-6]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musique_editlr1e-7]=2025-02-18_06-35-03_2633255081
    [llama3.2-1B_on_musique_editlr1e-8]=2025-02-18_15-01-06_0891593193
    # lr_lr no EOS
    [llama3.2-1B_on_musique_lrlr1e-2]=2025-02-18_10-54-20_5105925359
    [llama3.2-1B_on_musique_lrlr1e-3]=2025-02-17_16-11-33_8776393723
    [llama3.2-1B_on_musique_lrlr1e-4]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musique_lrlr1e-5]=2025-02-18_01-30-25_7593913958
    [llama3.2-1B_on_musique_lrlr1e-6]=2025-02-18_00-47-40_0516168634
)



n_val=1000
prompt=no
# task=zsre
# archive=2025-02-10_08-19-14_2641409766

for exp_dir_name in eos-sft_musique_propagator_p0
# for exp_dir_name in llama3.2-1B_on_zsre-full llama3.2-1B_on_musiqueQonly llama3.2-1B_on_zsre-14K llama3.2-1B_on_musiqueQ_w-eos
# exp_dir_name=""
do
archive=${name2id[$exp_dir_name]}

task=musique
# sft(q_p, a_p)
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False
# task=zsre
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False

python run_mend_edit_memorizer.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=2doc generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=True



done
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}-wICL/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=True

# exp_dir_name="llama3.2-1B_on_musiqueQonly"
# archive=${name2id[$exp_dir_name]}

# sft(q_p, a_p)
# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}-debug/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True


# exp_dir_name="llama3.2-1B_on_musiqueQonly"
# archive=${name2id[$exp_dir_name]}

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}-repPenalty/${task} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True

# python run_mend_edit.py +alg=mend +experiment=${task} +model=llama3.2-1B archive=${archive} eval_only=True generation.save_dir=exp_output/${exp_dir_name}-debug/${task} val_steps=20 edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True

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
