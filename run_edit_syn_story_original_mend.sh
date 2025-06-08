export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # metatrain debug on date data
    [synstory_original_mend_noshare_midupper3]=2025-05-06_18-23-35_443434257
    [synstory_original_mend_share_midupper3]=2025-05-15_00-58-32_426589717
    [synstory_original_mend_zsre_share_top3]=2025-05-07_16-59-26_3705393258
    [synstory_4K_noshare_midupper3_ablate_cpt]=2025-05-08_20-33-55_8610332217

    [synstory_original_mend_share_midupper3]=2025-05-15_00-58-32_426589717
    [synstory_4K_share_midupper3_ablate_cpt]=2025-05-15_17-23-43_2037225865
    # qwen2.5-1.5B
    [synstory_original_mend_zsre_share_top_qwen]=2025-06-01_02-51-32_3151285596
    [synstory_original_mend_share_midupper_qwen]=2025-06-01_01-08-18_6428861864
    [synstory_4K_share_midupper_ablate_cpt_qwen]=2025-06-02_06-07-49_8915405836
)


n_val=200
prompt=no
task=syn_story_mend

exp_dir_name=synstory_original_mend_share_midupper_qwen
archive=${name2id[$exp_dir_name]}

for date_data in profile # 4K_test_id 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation # 4K_test_id  # 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation  # profiling
do

# midupper 
# python run_original_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-midupper3 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True

# qwen2.5-1.5B
python run_original_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True

# top3
# python run_original_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-top3 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True


done
