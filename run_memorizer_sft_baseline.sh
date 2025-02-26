export CUDA_VISIBLE_DEVICES=2

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

input_format=2hop

# for epoch in 1 2
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
#     --save_dir_suffix="tune_num_epoch" \
#     --report_to="none" \

# done
# done



epoch=4

# for lr in 5e-6 2e-6 1e-6
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

lr=1e-5

for input_format in two-1hop # first-1hop second-1hop
do 

for example_idx in 7 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 379 380 417 509 510 518 520 521 532 545 658 853 866 889 890 893 894 896 897 900 901 963 975 984 986
do

echo "Example idx: ${example_idx}"

accelerate launch --config_file="fsdp_config.yaml" \
    --main_process_port 29900 \
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