#!/bin/bash
#SBATCH -J edit-qwen-max-syn-story       # Job name
#SBATCH -o slurm-outputs/%x.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/%x.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1 
#SBATCH -t 6:00:00        # Run time (hh:mm:ss)
#SBATCH -A CCR25005       # Allocation name (req'd if you have more than 1)

export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    [qwen_noshare_max_13_27]=2025-05-09_22-38-52_2439112622
    [qwen_share_max_13_27]=2025-05-15_00-48-06_8483828866

    [qwen_noshare_max_4K_14_27]=2025-06-02_02-17-48_9852067005
    [qwen_share_max_4K_14_27]=2025-06-02_02-17-56_8514513907
    [qwen_noshare_max_30K_14_27]=2025-06-02_01-54-37_0645664839
    [qwen_share_max_30K_14_27]=2025-06-02_02-18-16_4099521066

    [qwen_share_max_30K_14_27_lr1e-6]=2025-06-09_11-16-48_0488543185
)


n_val=500
prompt=no
task=syn_story

exp_dir_name=qwen_share_max_30K_14_27_lr1e-6
archive=${name2id[$exp_dir_name]}

for date_data in 4K_test_id 4K_test_ood-entity 4K_test_ood-relation 4K_test_ood 
do

python run_mend_edit_syn_story_qwen.py +alg=mend +experiment=${task} +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-max archive="${archive}" eval_only=True generation.save_dir=synstory_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=False +date_data=${date_data} mend.shared=True +exp_name=${exp_dir_name} 

done