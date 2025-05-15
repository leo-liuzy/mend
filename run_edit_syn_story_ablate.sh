export CUDA_VISIBLE_DEVICES=5

declare -A name2id=(
    # ablation
    [4K_heavy_noshare_top3]=2025-05-08_18-05-49_1289175700
    [4K_heavy_share_midupper3]=2025-05-08_18-04-57_3311174838
)


n_val=500
prompt=no
task=syn_story



date_data="4K_test_ood"

for date_data in 4K_test_id # 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation 
do
# exp_dir_name=4K_heavy_noshare_top3
# archive=${name2id[$exp_dir_name]}
# # top 3
# python run_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-top3 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=False +exp_name=${exp_dir_name} 

exp_dir_name=4K_heavy_share_midupper3
archive=${name2id[$exp_dir_name]}
# sharing
python run_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-midupper3 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=True +exp_name=${exp_dir_name} 

done