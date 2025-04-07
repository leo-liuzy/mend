export CUDA_VISIBLE_DEVICES=4

declare -A name2id=(
    # metatrain debug on date data
    [common-country_3K_heavy_share_top3]=2025-03-22_19-38-18_2276916953
    [common-country_3K_heavy_share_mid-upper3]=2025-03-22_19-37-37_8161358232

    [common-country_3K_heavy_noshare_top3]=2025-03-22_19-48-59_4148028265
    [common-country_3K_heavy_noshare_mid-upper3]=2025-03-22_19-48-59_3832569414

    [common-country_3K_light_share_top3]=2025-03-22_19-42-08_988540734
    [common-country_3K_light_share_mid-upper3]=2025-03-22_19-42-09_0177528181

    [common-country_3K_light_noshare_top3]=2025-03-23_01-29-39_632768923
    [common-country_3K_light_noshare_mid-upper3]=2025-03-22_19-49-35_6412001835
)


n_val=100
prompt=no
task=country_syn

exp_dir_name=common-country_3K_light_share_mid-upper3
archive=${name2id[$exp_dir_name]}

for date_data in all_propagation_ood all_propagation_ood_w_ood_country
do

python run_mend_edit_country_ood.py +alg=mend +experiment=${task} +model=llama3.2-1B-common-country-eos-sft-mid-upper archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} # mend.shared=False

# python run_mend_edit_country_ood.py +alg=mend +experiment=${task} +model=llama3.2-1B-common-country-eos-sft archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=False

done
