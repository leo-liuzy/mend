export CUDA_VISIBLE_DEVICES=1

declare -A name2id=(
    # metatrain debug on date data
    [3K_heavy_noshare_mid-upper3_template-1_seen-50]=2025-04-14_00-28-01_9392896490
    [3K_heavy_noshare_mid-upper3_template-1_seen-100]=2025-04-14_00-28-11_0027379133
    [3K_heavy_noshare_mid-upper3_template-1_seen-200]=2025-04-14_05-14-31_8442247340
    [3K_heavy_noshare_mid-upper3_template-1_seen-350]=2025-04-14_00-23-48_6803551272

    [3K_heavy_noshare_mid-upper3_template-2_seen-50]=2025-04-14_04-39-40_8926335243
    [3K_heavy_noshare_mid-upper3_template-2_seen-350]=2025-04-14_04-57-54_1071293484

    [3K_heavy_noshare_mid-upper3_template-5_seen-50]=2025-04-14_09-09-22_8860473327
    [3K_heavy_noshare_mid-upper3_template-5_seen-350]=2025-04-14_08-31-26_0851602480

    # [3K_heavy_noshare_mid-upper3_template-7_seen-50]=2025-04-14_14-30-52_0799769634
    [3K_heavy_noshare_mid-upper3_template-7_seen-100]=2025-04-14_08-49-11_3649137932
    # [3K_heavy_noshare_mid-upper3_template-7_seen-200]=2025-04-14_15-31-09_8567314752
    [3K_heavy_noshare_mid-upper3_template-7_seen-350]=2025-04-14_13-06-58_256753527
)


n_val=100
prompt=no
task=country_syn_v2

exp_dir_name=3K_heavy_noshare_mid-upper3_template-7_seen-350
archive=${name2id[$exp_dir_name]}

for date_data in seen_pair_seen_country unseen_pair_seen_country unseen_pair_unseen_country # _ood
do

python run_mend_edit_country_v2.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-country-template-format-lr2e-6-mid-upper archive=${archive} eval_only=True generation.save_dir=country_exp_out/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=False +exp_name=${exp_dir_name} 

# python run_mend_edit_country.py +alg=mend +experiment=${task} +model=llama3.2-1B-common-country-eos-sft archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} # mend.shared=False


done
