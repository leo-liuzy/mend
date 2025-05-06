export CUDA_VISIBLE_DEVICES=1

declare -A name2id=(
    # metatrain debug on date data
    [ripple_edits_heavy-share-top3]=2025-03-24_02-11-22_5014004840
    [ripple_edits_heavy-share-mid-upper3]=2025-03-24_02-14-44_4863903024
    [ripple_edits_heavy-noshare-top3]=2025-03-24_02-13-02_8204171418
    [ripple_edits_heavy-noshare-mid-upper3]=2025-03-24_02-13-39_2649233719
)


n_val=200 # 50
prompt=no
task=ripple_edits

# exp_dir_name=ripple_edits_heavy-noshare-mid-upper3
# archive=${name2id[$exp_dir_name]}

# config_name=llama3.2-1B-eos-sft-top 
date_data=all

# mom2_dataset="wikipedia"

for config_name in llama3.2-1B-eos-sft-top  # llama3.2-1B-eos-sft-mid-upper # llama3.2-1B-eos-sft-top # llama3.2-1B-eos-sft-mid-upper
do
for mom2_dataset in "ripple_all" 
do

# python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} # mend.shared=False

python run_memit_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid-upper eval_only=True generation.save_dir=ripple_exp_output/${config_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} +config_name=${config_name} +mom2_dataset=${mom2_dataset} # mend.shared=False

done
done
