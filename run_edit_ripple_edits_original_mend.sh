export CUDA_VISIBLE_DEVICES=2

declare -A name2id=(
    # metatrain debug on date data
    [ripple_edits_recent_original_mend]=2025-03-30_00-31-26_5729277951
)


n_val=150
prompt=no
task=ripple_edits

exp_dir_name=ripple_edits_recent_original_mend
archive=${name2id[$exp_dir_name]}

for date_data in recent
do

# python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False

python run_original_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid-upper archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=sft edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False


done
