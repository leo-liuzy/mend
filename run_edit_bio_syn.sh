export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # metatrain debug on date data
    [common-date-year-after_1K]=2025-03-15_18-54-39_5970343765
    [common-date-year-after_1K_noshare]=2025-03-15_18-59-31_7549442309
    
    [common-date-year-after_10K]=2025-03-15_18-55-51_1348523953

    [common-date-year-after_100K]=2025-03-15_18-56-25_1434749561
    [common-date-year-after_100K_noshare]=2025-03-15_18-57-35_6230462214
)


n_val=100
prompt=no
task=bio_syn

exp_dir_name=common-date-year-after_100K_noshare
archive=${name2id[$exp_dir_name]}

for date_data in n n+1
do
python run_mend_edit_bio_syn.py +alg=mend +experiment=${task} +model=llama3.2-1B-common-date-year-after-eos-sft archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=False
done
