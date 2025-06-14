export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    [qwen_noshare_max_13_27]=2025-05-09_22-38-52_2439112622
    [qwen_share_max_13_27]=2025-05-15_00-48-06_8483828866

    # [qwen_noshare_max_14_27]=
    # [qwen_share_max_14_27]=
    [qwen_noshare_max]=2025-05-09_22-38-52_2439112622
    [qwen_share_max]=2025-05-15_00-48-06_8483828866
    [qwen_share_midupper]=2025-05-31_17-45-19_4706733892
    [qwen_share_top]=2025-06-01_02-55-50_2230889828

    [qwen_share_max_4K_14_27]=2025-06-02_02-17-56_8514513907
    [qwen_4K_paraphrase_share_midupper]=2025-06-01_17-32-32_6805682344
)


n_val=500
prompt=no
task=syn_story

exp_dir_name=qwen_share_max_4K_14_27
archive=${name2id[$exp_dir_name]}

for date_data in profile # 4K_test_id 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation
do

# python run_mend_edit_syn_story_qwen.py +alg=mend +experiment=${task} +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10 archive="${archive}" eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=True +exp_name=${exp_dir_name} 

python run_mend_edit_syn_story_qwen.py +alg=mend +experiment=${task} +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-max archive="${archive}" eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=True +exp_name=${exp_dir_name} 

done