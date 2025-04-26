#!/bin/bash
#SBATCH -J eval_know_prop       # Job name
#SBATCH -o slurm-outputs/%x.o%j       # Name of stdout output file
#SBATCH -e slurm-outputs/%x.e%j       # Name of stderr output file
#SBATCH -p gh-dev          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes
##SBATCH --ntasks-per-node=1 
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH -A CCR25005       # Allocation name (req'd if you have more than 1)

export CUDA_VISIBLE_DEVICES=0

declare -A name2id=(
    # Final
    [ripple_edits_all_heavy-noshare-all-in-outer_10-15]=2025-04-24_22-17-07_8961755309
    [ripple_edits_all_heavy-noshare-all-in-outer_7-12]=2025-04-24_22-20-17_4805768358
    [ripple_edits_all_heavy-noshare-all-in-outer_9-14]=2025-04-24_22-31-48_3849324322
    [ripple_edits_all_heavy-noshare-all-in-outer_8-13]=2025-04-25_00-42-11_0847377006
    [ripple_edits_all_heavy-noshare-all-in-outer_9-15]=2025-04-24_22-25-12_2471377612
    [ripple_edits_all_heavy-noshare-all-in-outer_7-15]=2025-04-24_22-53-55_5879688392
    [ripple_edits_all_heavy-noshare-all-in-outer_4-15]=2025-04-24_22-53-49_0166002514
    [ripple_edits_all_heavy-noshare-all-in-outer_7-13]=2025-04-25_02-50-02_4743999764
    [ripple_edits_all_heavy-noshare-all-in-outer_7-14]=2025-04-25_04-56-28_2025436588

    # share
    [ripple_edits_all_heavy-share-all-in-outer_7-12]=2025-04-26_13-07-40_612784232
    [ripple_edits_all_heavy-share-all-in-outer_7-13]=2025-04-26_12-53-57_3860445861
    [ripple_edits_all_heavy-share-all-in-outer_7-14]=2025-04-26_13-06-03_655851344
    [ripple_edits_all_heavy-share-all-in-outer_8-13]=2025-04-26_12-50-28_4555958376
)

declare -A name2config=(
    [ripple_edits_all_heavy-noshare-all-in-outer_7-12]=llama3.2-1B-eos-sft-7-12
    [ripple_edits_all_heavy-noshare-all-in-outer_7-13]=llama3.2-1B-eos-sft-7-13
    [ripple_edits_all_heavy-noshare-all-in-outer_7-14]=llama3.2-1B-eos-sft-7-14
    [ripple_edits_all_heavy-noshare-all-in-outer_7-15]=llama3.2-1B-eos-sft-7-15
    [ripple_edits_all_heavy-noshare-all-in-outer_8-13]=llama3.2-1B-eos-sft-8-13
    [ripple_edits_all_heavy-noshare-all-in-outer_9-14]=llama3.2-1B-eos-sft-9-14
    [ripple_edits_all_heavy-noshare-all-in-outer_9-15]=llama3.2-1B-eos-sft-9-15
    [ripple_edits_all_heavy-noshare-all-in-outer_10-15]=llama3.2-1B-eos-sft-10-15
    [ripple_edits_all_heavy-noshare-all-in-outer_4-15]=llama3.2-1B-eos-sft-4-15

    # share
    [ripple_edits_all_heavy-share-all-in-outer_7-12]=llama3.2-1B-eos-sft-7-12
    [ripple_edits_all_heavy-share-all-in-outer_7-13]=llama3.2-1B-eos-sft-7-13
    [ripple_edits_all_heavy-share-all-in-outer_7-14]=llama3.2-1B-eos-sft-7-14
    [ripple_edits_all_heavy-share-all-in-outer_8-13]=llama3.2-1B-eos-sft-8-13
)


prompt=no
task=ripple_edits

# exp_dir_name=ripple_edits_recent+popular+random_heavy-noshare-mid-upper3
date_data=all #_wo_random
n_val=200

for exp_dir_name in ripple_edits_all_heavy-share-all-in-outer_7-14 #  ripple_edits_all_heavy-noshare-all-in-outer_7-12 ripple_edits_all_heavy-noshare-all-in-outer_7-13 ripple_edits_all_heavy-noshare-all-in-outer_7-14 #  ripple_edits_all_heavy-noshare-all-in-outer_9-15 ripple_edits_all_heavy-noshare-all-in-outer_10-15 ripple_edits_all_heavy-noshare-all-in-outer_4-15  #  ripple_edits_all_heavy-noshare-all-in-outer_7-15 ripple_edits_all_heavy-noshare-all-in-outer_8-13  #  #   #  # 
do

archive=${name2id[$exp_dir_name]}
config=${name2config[$exp_dir_name]}
# python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=True mend.n_hidden=2

nohup python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=${config} archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False > ${exp_dir_name}.log 2>&1 & # mend.rank=240

# python run_mend_edit_ripple_edits.py +alg=mend +experiment=${task} +model=llama3.2-1B-eos-sft-mid2 archive=${archive} eval_only=True generation.save_dir=ripple_exp_output/${exp_dir_name}/${task} val_steps=${n_val} edit_loss=clm edit_input=seen generation.prompt=${prompt} +do_generation=True +add_bos=True +add_eos=True +add_eos_accuracy=True +gen_w_bos=True +add_icl=False +spec_question=True +date_data=${date_data} mend.shared=False 

done
