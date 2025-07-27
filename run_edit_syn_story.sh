export CUDA_VISIBLE_DEVICES=2

declare -A name2id=(
    [4K_heavy_noshare_top3]=2025-05-08_18-04-57_3311174838

    [4K_heavy_noshare_midupper3]=2025-05-06_01-39-29_0943027568
    [30K_heavy_noshare_midupper3]=2025-05-06_11-24-32_1300987290
    [4K_heavy_noshare_layer4_15]=2025-05-07_06-16-18_6346127998
    [4K_paraphrase_noshare_midupper3]=2025-05-08_20-29-37_0233723668

    [30K_heavy_share_midupper3]=2025-05-21_10-55-13_9117923049
)


n_val=500
prompt=no
task=syn_story

exp_dir_name=4K_heavy_noshare_layer4_15
archive=${name2id[$exp_dir_name]}
# date_data="4K_test_ood"

for date_data in 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation 4K_test_id
do
# sum(p.numel() for p in alg.mend.parameters())
# python run_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-midupper3 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=True +exp_name=${exp_dir_name} 

python run_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=False +exp_name=${exp_dir_name} 

done