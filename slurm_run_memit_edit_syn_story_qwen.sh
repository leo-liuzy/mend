#!/bin/bash
#SBATCH -J memit-wiki-qwen       # Job name
#SBATCH -o slurm-outputs/%x.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/%x.e%j       # Name of stderr output file
#SBATCH -p gh          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1 
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
#SBATCH -A CCR25005       # Allocation name (req'd if you have more than 1)

export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # metatrain debug on date data
    [ripple_edits_heavy-share-top3]=2025-03-24_02-11-22_5014004840
    [ripple_edits_heavy-share-mid-upper3]=2025-03-24_02-14-44_4863903024
    [ripple_edits_heavy-noshare-top3]=2025-03-24_02-13-02_8204171418
    [ripple_edits_heavy-noshare-mid-upper3]=2025-03-24_02-13-39_2649233719
)


n_val=200 # 50
prompt=no
task=syn_story

# exp_dir_name=ripple_edits_heavy-noshare-mid-upper3
# archive=${name2id[$exp_dir_name]}

# config_name=llama3.2-1B-eos-sft-top 


mom2_dataset="wikipedia"

# mom2_dataset="synstory_4K"

for config_name in qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-estimated-wiki
do

for date_data in 4K_test_id 4K_test_ood 4K_test_ood-entity 4K_test_ood-relation # "4K_test_ood" # "4K_test_id" "4K_test_ood-entity"  "4K_test_ood" # "4K_test_id" "4K_test_ood" # 
do

python run_memit_edit_syn_story.py +alg=mend +experiment=${task} +model=qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10 eval_only=True generation.save_dir=synstory_exp_output/${config_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} +config_name=${config_name} +mom2_dataset=${mom2_dataset}

done
done
# llama3.2-1B-eos-sft-template-format-curated-v1-lr2e-6-sample-10-midupper3