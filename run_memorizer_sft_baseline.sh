export CUDA_VISIBLE_DEVICES=6

export WANDB_MODE=online

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
bs=1
per_device_train_batch_size=1
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
max_seq_length=256
epoch=4
# max_steps=1

lr=1e-5

for input_format in 2hop # two-1hop 2hop second-1hop first-1hop # two-1hop # first-1hop second-1hop
do 

for example_idx in 172 # {0..999}
do

echo "Example idx: ${example_idx}"

accelerate launch --config_file="fsdp_config.yaml" \
    --main_process_port 29500 \
    memorizer_sft_baseline.py \
    --seed=${seed} \
    --output_dir="${PWD}/models" \
    --learning_rate=${lr} \
    --lr_scheduler_type=constant \
    --weight_decay=${weight_decay} \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --gradient_accumulation_steps=${grad_acc} \
    --max_seq_length=${max_seq_length} \
    --max_grad_norm=${max_grad_norm} \
    --optim="adamw_torch" \
    --dataset_text_field="text" \
    --bf16=True \
    --eval_strategy="no" \
    --save_strategy="no" \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=1 \
    --report_to="wandb" \
    --num_train_epochs=${epoch} \
    --run_name="memorizer-sft-baseline" \
    --input_format=${input_format} \
    --example_idx=${example_idx} \
    --report_to="none" \
    --spec_question=True

done
done



# epoch=4
# for lr in 1e-5 5e-6 2e-6 1e-6
# do
# for example_idx in {0..99}
# do
# echo "Example idx: ${example_idx}"

# accelerate launch --config_file="fsdp_config.yaml" \
#     --main_process_port 29600 \
#     memorizer_sft_baseline.py \
#     --seed=${seed} \
#     --output_dir="${PWD}/models" \
#     --learning_rate=${lr} \
#     --lr_scheduler_type=constant \
#     --weight_decay=${weight_decay} \
#     --per_device_train_batch_size=${per_device_train_batch_size} \
#     --gradient_accumulation_steps=${grad_acc} \
#     --max_seq_length=${max_seq_length} \
#     --max_grad_norm=${max_grad_norm} \
#     --optim="adamw_torch" \
#     --dataset_text_field="text" \
#     --bf16=True \
#     --eval_strategy="no" \
#     --save_strategy="no" \
#     --logging_strategy="steps" \
#     --logging_first_step=True \
#     --logging_steps=1 \
#     --report_to="wandb" \
#     --num_train_epochs=${epoch} \
#     --run_name="memorizer-sft-baseline" \
#     --input_format=${input_format} \
#     --example_idx=${example_idx} \
#     --save_dir_suffix="tune_lr" \
#     --report_to="none" \

# done
# done