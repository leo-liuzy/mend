export CUDA_VISIBLE_DEVICES=0

n_val=100
task=musique
prompt=no


base_model_name=curator


# sft(q_p, a_p)
ice=True

for date_data in 4K_test_id 4K_test_ood 4K_test_ood-relation 4K_test_ood-entity
do

python run_base_generate_synstory_curator.py +alg=mend +experiment=${task} eval_only=True generation.save_dir=synstory_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=${ice} +date_data=${date_data} 

done
# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common

# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common_train


# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=False +date_data=country_syn_ood_v2