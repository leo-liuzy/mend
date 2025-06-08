export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(

    [ripple_edits_all_heavy-noshare-mid-upper3]=2025-04-21_12-40-48_5366252073
)



n_val=100
task=musique
prompt=no
# task=zsre
# archive=2025-02-10_08-19-14_2641409766
exp_dir_name="ripple_edits_all_heavy-noshare-mid-upper3"
archive=${name2id[$exp_dir_name]}

# base_model_name=llama3.2-1B-instruct
# base_model_name=llama3.2-1B-common-country-eos-sft-country_syn-pretrain-top3
# base_model_name=llama3.2-1B-common-country-eos-sft # country_syn-pretrain-all
# base_model_name=llama3.2-1B-common-country-eos-sft
# base_model_name=llama3.2-1B-eos-sft-country-template-format-lr1e-6
base_model_name=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10
# base_model_name=llama3.2-1B-eos-sft
# base_model_name=llama3.2-1B
# base_model_name=qwen2.5-1.5B
# sft(q_p, a_p)

python run_base_generate_synstory_qwen.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=False +date_data=profile
# icl=False

# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common

# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common_train


# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=False +date_data=country_syn_ood_v2