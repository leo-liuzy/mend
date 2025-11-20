export CUDA_VISIBLE_DEVICES=2

declare -A name2id=(
    [4K_noshare_layer4_15_instruct_resurface]=2025-10-28_22-43-57_7391638448

)


n_val=50
prompt=no
task=syn_story_instruct_paraphrase

exp_dir_name=4K_noshare_layer4_15_instruct_resurface
archive=${name2id[$exp_dir_name]}
# date_data="4K_test_ood"

for date_data in 4K_test_id
do

python run_mend_edit_syn_story_instruct_resurface.py +alg=mend +experiment=${task} +model=qwen2.5-1.5B-instruct archive=${archive} eval_only=True generation.save_dir=/u/zliu/datastor1/LLaMA-Factory/synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=True +exp_name=${exp_dir_name} 

# python run_mend_edit_syn_story.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-4-15 archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=False +exp_name=${exp_dir_name} 

done