export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # metatrain debug on date data
    [common-date-year-after_3K_heavy_share_top3]=2025-03-20_18-55-20_8518563732
    [common-date-year-after_3K_heavy_share_mid-upper3]=2025-03-20_18-59-01_26816292

    [common-date-year-after_3K_heavy_noshare_top3]=2025-03-20_19-01-11_3331056858
    [common-date-year-after_3K_heavy_noshare_mid-upper3]=2025-03-20_19-08-53_1821694575

    [common-date-year-after_3K_light_share_top3]=2025-03-20_20-29-05_6288825978
    [common-date-year-after_3K_light_share_mid-upper3]=2025-03-20_20-29-05_622543347

    [common-date-year-after_3K_light_noshare_top3]=2025-03-21_10-44-03_8000446978
    [common-date-year-after_3K_light_noshare_mid-upper3]=2025-03-21_10-47-20_8168889024
)


n_val=100
prompt=no
task=bio_syn_v2

exp_dir_name=common-date-year-after_3K_heavy_noshare_top3
archive=${name2id[$exp_dir_name]}

for date_data in all_propagation
do
python run_mend_edit_bio_syn_v2.py +alg=mend +experiment=${task} +model=llama3.2-1B-common-date-year-after-eos-sft archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False
done
