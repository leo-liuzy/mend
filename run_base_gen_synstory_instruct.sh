#!/bin/bash
#SBATCH -J gen-only       # Job name
#SBATCH -o slurm-outputs/%x.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/%x.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1
#SBATCH -t 6:00:00        # Run time (hh:mm:ss)
#SBATCH -A CCR25005       # Allocation name (req'd if you have more than 1)

export CUDA_VISIBLE_DEVICES=0

n_val=100
task=musique
prompt=no


# base_model_name=llama3.2-1B-instruct
base_model_name=llama3.2-3B-instruct
# base_model_name=qwen2.5-1.5B-instruct
# base_model_name=qwen2.5-7B-instruct
# base_model_name=qwen2.5-32B-instruct


# sft(q_p, a_p)
ice=True
# profiling 4K_test_ood 4K_test_ood-relation 4K_test_ood-entity 4K_test_id
for date_data in 4K_test_id
do

python run_base_generate_synstory_instruct.py +alg=mend +experiment=${task} +model=${base_model_name} eval_only=True generation.save_dir=synstory_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=${ice} +date_data=${date_data} 

done
# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common

# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=${icl} +ice=False +date_data=common_train


# python run_base_generate_country_ood.py +alg=mend +experiment=${task} +model=${base_model_name} archive=${archive} eval_only=True generation.save_dir=debug_exp_output/${base_model_name} val_steps=${n_val} edit_loss=sft edit_input=question generation.prompt=${prompt} +do_generation=True +add_eos=True +gen_w_bos=True +add_icl=False +ice=False +date_data=country_syn_ood_v2