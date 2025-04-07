export CUDA_VISIBLE_DEVICES=1

declare -A name2id=(
    # metatrain debug on date data
    [drop-6K_heavy_share_mid-upper3]=2025-03-30_23-44-40_9347648815
)


n_val=100
prompt=no
task=drop

exp_dir_name=drop-6K_heavy_share_mid-upper3
archive=${name2id[$exp_dir_name]}

for date_data in drop
do

python run_mend_edit_drop.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid-upper archive=${archive} eval_only=True generation.save_dir=drop_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} # mend.shared=False

done
