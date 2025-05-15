export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    [3B_4K_heavy_noshare_midupper3]=2025-05-11_00-22-02_5924024672 # this is max not midupper
    [3B_4K_heavy_share_max]=2025-05-15_00-48-07_4674102570
)


n_val=500
prompt=no
task=syn_story

exp_dir_name=3B_4K_heavy_share_max
archive=${name2id[$exp_dir_name]}
# date_data="4K_test_ood"

for date_data in 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation 4K_test_id # 4K_test_ood 4K_test_ood-entity #  4K_test_ood-relation 4K_test_id
do

python run_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-3B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-max archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=True +exp_name=${exp_dir_name} 

done