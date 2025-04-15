export CUDA_VISIBLE_DEVICES=2

declare -A name2id=(
    # metatrain debug on date data
    [ripple_edits_heavy-share-top3]=2025-03-24_02-11-22_5014004840
    [ripple_edits_heavy-share-mid-upper3]=2025-03-24_02-14-44_4863903024
    [ripple_edits_heavy-noshare-top3]=2025-03-24_02-13-02_8204171418
    [ripple_edits_heavy-noshare-mid-upper3]=2025-03-24_02-13-39_2649233719

    [ripple_edits_recent_heavy-share-top3]=2025-03-27_21-25-59_3900034961
    [ripple_edits_recent_heavy-share-mid-upper3]=2025-03-27_20-45-47_1361681194
    [ripple_edits_recent_heavy-noshare-top3]=2025-03-28_20-59-37_156295867
    [ripple_edits_recent_heavy-noshare-mid-upper3]=2025-03-27_20-46-19_2019788024

    # Hyper tuning
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_rank2880]=2025-04-13_03-27-01_1775775785
    [ripple_edits_recent+popular_heavy-share-top3_nhidden2]=2025-04-13_03-23-19_8708526191
    [ripple_edits_recent+popular_heavy-share-top3_nhidden4]=2025-04-13_16-57-02_4262358989

)


n_val=150
prompt=no
task=ripple_edits

exp_dir_name=ripple_edits_recent+popular_heavy-share-top3_nhidden2
archive=${name2id[$exp_dir_name]}

for date_data in recent+popular
do

python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True

# python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid-upper archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False


done
