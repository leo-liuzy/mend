export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    [llama3.2-1B_on_zsre-full]=2025-02-10_08-19-14_2641409766
    [llama3.2-1B_on_zsre-14K]=2025-02-17_16-06-54_298004526
    [llama3.2-1B_on_musiqueQonly]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musiqueQ_w-eos]=2025-02-20_21-05-21_5456934010
    # metatrain with 2 doc
    [musique_propagator_p0]=2025-02-22_06-58-48_2302322883
    # [musique_propagator_p0_w-newline]=2025-02-22_21-40-54_1897999909
    [musique_propagator_p0_w-newline_icl]=2025-02-22_21-40-16_9068188311
    # edit_lr no EOS
    [llama3.2-1B_on_musique_editlr1e-4]=2025-02-18_00-48-02_2245122483
    [llama3.2-1B_on_musique_editlr1e-5]=2025-02-17_15-56-35_0163092923
    [llama3.2-1B_on_musique_editlr1e-6]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musique_editlr1e-7]=2025-02-18_06-35-03_2633255081
    [llama3.2-1B_on_musique_editlr1e-8]=2025-02-18_15-01-06_0891593193
    # lr_lr no EOS
    [llama3.2-1B_on_musique_lrlr1e-2]=2025-02-18_10-54-20_5105925359
    [llama3.2-1B_on_musique_lrlr1e-3]=2025-02-17_16-11-33_8776393723
    [llama3.2-1B_on_musique_lrlr1e-4]=2025-02-11_23-27-06_6306217737
    [llama3.2-1B_on_musique_lrlr1e-5]=2025-02-18_01-30-25_7593913958
    [llama3.2-1B_on_musique_lrlr1e-6]=2025-02-18_00-47-40_0516168634

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
base_model_name=llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10
# base_model_name=llama3.1-8B-eos-sft-template-format-curated-v1-lr2e-6-sample-10
# base_model_name=llama3.2-3B-eos-sft-template-format-curated-v1-lr2e-6-sample-10
# base_model_name=llama3.2-1B-eos-sft
# base_model_name=llama3.2-1B
# base_model_name=qwen2.5-1.5B

# sft(q_p, a_p)
ice=True

for date_data in profiling
do

python run_base_generate_synstory.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=synstory_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=${ice} +date_data=${date_data} 

done
# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common

# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common_train


# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=False +date_data=country_syn_ood_v2