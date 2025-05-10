export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    [qwen_noshare_max]=2025-05-09_22-38-52_2439112622
)


n_val=500
prompt=no
task=syn_story

exp_dir_name=qwen_noshare_max
archive=${name2id[$exp_dir_name]}

for date_data in 4K_test_ood-relation 4K_test_id # 4K_test_ood 4K_test_ood-entity 
do

python run_mend_edit_syn_story_qwen.py +alg=mend +experiment=${task} +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-max archive="${archive}" eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=False +exp_name=${exp_dir_name} 

done