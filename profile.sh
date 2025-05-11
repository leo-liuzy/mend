nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 > gpu_profile.csv &
NVIDIA_PID=$!

# Record start time
STARTTIME=$(date +%s)

# Run the actual script

# Base + Prepend
# bash run_base_gen_synstory.sh

# CPT
# bash run_clm_base_synstory.sh

# MEND
# bash run_edit_syn_story_original_mend.sh

# PropaMEND
bash run_edit_syn_story.sh

# Record end time
ENDTIME=$(date +%s)

# Stop monitoring
kill $NVIDIA_PID

# Calculate runtime
echo "Script runtime: $(($ENDTIME - $STARTTIME)) seconds"
echo "GPU usage data saved in gpu_profile.csv"
cat gpu_profile.csv | awk -F, '{print $2}' | sed 's/ MiB//' | sort -n | tail -1 | awk '{print $1 " MiB"}'
