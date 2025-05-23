export CUDA_VISIBLE_DEVICES=0

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

    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_rank2880]=2025-04-15_04-10-57_5518419406
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_rank1440]=2025-04-15_20-02-56_834455400
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_rank960]=2025-04-16_00-37-46_4857465816
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_rank480]=2025-04-16_13-34-53_0685846321
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_rank240]=2025-04-16_15-14-14_6736382614
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_rank120]=2025-04-16_22-52-33_1479932232

    [ripple_edits_recent+popular_heavy-noshare-mid4-upper3]=2025-04-15_04-18-04_2043417416
    [ripple_edits_recent+popular_heavy-noshare-mid4-lower3]=2025-04-15_09-06-24_0162268509
    [ripple_edits_recent+popular_heavy-noshare-mid-lower3]=2025-04-15_11-08-41_0907547834
    [ripple_edits_recent+popular_heavy-noshare-bottom3]=2025-04-15_16-15-56_2319489525

    [ripple_edits_recent+popular_heavy-share-top3_nhidden2]=2025-04-15_01-12-41_8448543075
    [ripple_edits_recent+popular_heavy-share-top3_nhidden4]=2025-04-15_04-44-05_793796871

    ## random seed
    [ripple_edits_heavy-noshare-mid-upper3-seed1]=2025-04-16_00-00-36_0010211627
    [ripple_edits_heavy-noshare-mid-upper3-seed2]=2025-04-16_07-24-51_8301223200

    # outerloop
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_all-in-outerloop]=2025-04-16_14-46-50_8956486717
    # Loss 
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_cbase0]=2025-04-16_18-39-23_9597674851
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_cbase02]=2025-04-17_02-37-34_9773806917
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_cbase04]=2025-04-17_10-09-23_7087907152
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_cbase06]=2025-04-17_02-38-56_9171446672
    [ripple_edits_recent+popular_heavy-noshare-mid-upper3_cbase08]=2025-04-16_18-39-53_707494128

    # data augmentation
    [ripple_edits_recent+popular+random_heavy-noshare-mid-upper3]=2025-04-15_21-10-24_982004503
    [ripple_edits_recent+popular+ekp_heavy-noshare-mid-upper3]=2025-04-16_23-24-50_1778685779
    [ripple_edits_recent+popular+recoe_heavy-noshare-mid-upper3]=2025-04-16_21-52-46_0156755258

    # Final
    [ripple_edits_all_heavy-noshare-mid-upper3_all-in-outer]=2025-04-21_12-40-08_4547596845
    [ripple_edits_all_heavy-noshare-mid-upper3]=2025-04-21_12-40-48_5366252073
    [ripple_edits_all_recent+popular_heavy-noshare-mid-upper3_all-in-outer]=2025-04-21_17-05-43_6927273155
    [ripple_edits_all_recent+popular_heavy-noshare-mid-upper3]=2025-04-21_17-00-34_8850918366
)




prompt=no
task=ripple_edits

# exp_dir_name=ripple_edits_recent+popular+random_heavy-noshare-mid-upper3
date_data=all #_wo_random
n_val=200

for exp_dir_name in ripple_edits_all_recent+popular_heavy-noshare-mid-upper3
do

archive=${name2id[$exp_dir_name]}
# python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True mend.n_hidden=2

python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid-upper archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False # mend.rank=240

# python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid2 archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False 

done
