export CUDA_VISIBLE_DEVICES=1

declare -A name2id=(
    # metatrain debug on date data
    [ripple_edits_recent_original_mend]=2025-03-30_00-31-26_5729277951
    [ripple_edits_recent+popular_original_mend]=2025-04-15_01-08-39_2385105405
    [ripple_edits_recent+popular_original_mend_share_top3]=2025-04-15_19-08-58_7068849019
    [zereFull_original_mend_noshare_midupper3]=2025-04-15_19-05-14_7051032117
    [zereFull_original_mend_share_top3]=2025-04-15_19-04-46_7645818205

    [ripple_edits_all_original_mend_share_top3]=2025-04-22_05-09-26_9287821869
    [ripple_edits_all_original_mend_noshare_midupper3]=2025-04-22_05-10-18_2495767514
    [ripple_edits_all_original_mend_share_midupper3]=2025-05-15_01-13-59_1708014893
    
)


n_val=200
prompt=no
task=ripple_edits

exp_dir_name=ripple_edits_all_original_mend_share_midupper3
archive=${name2id[$exp_dir_name]}

for date_data in all
do

# python run_original_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True

python run_original_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid-upper archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True


done
