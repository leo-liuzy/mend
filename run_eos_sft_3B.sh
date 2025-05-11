export CUDA_VISIBLE_DEVICES=0 #,1,2,3 # 4,5,6,7

export WANDB_MODE=online

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
eos_sft=False

if [ "$eos_sft" = True ]; then
    # this is for eos-sft
    bs=128
    per_device_train_batch_size=16
    lr=1e-5 # this is for eos-sft
    output_dir=models/Llama-3.2-3B-eos-sft
else
    # this is for template-sft
    bs=10 
    per_device_train_batch_size=2
    lr=2e-6 
    output_dir=models/Llama-3.2-3B-eos-sft-template-format-curated-v1-lr${lr}-sample-10
fi


# this is for template-sft
# bs=10
# per_device_train_batch_size=2
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
warmup_ratio=0.03
max_seq_length=128
epoch=2
# max_steps=1

# lr=2e-6
# lr=1e-5 # this is for eos-sft

# output_dir=models/Llama-3.1-8B-eos-sft
# output_dir=models/Llama-3.1-8B-eos-sft-template-format-curated-v1-lr${lr}-sample-10

# accelerate launch --config_file="fsdp_config.yaml" \
#     --main_process_port 29601 \

python lightweight_sft_3B.py \
    --output_dir="${output_dir}" \
    --seed=${seed} \
    --learning_rate=${lr} \
    --lr_scheduler_type=linear \
    --weight_decay=${weight_decay} \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=${grad_acc} \
    --max_seq_length=${max_seq_length} \
    --max_grad_norm=${max_grad_norm} \
    --optim="adamw_torch" \
    --dataset_text_field="text" \
    --lr_scheduler_type="linear" \
    --warmup_ratio=${warmup_ratio} \
    --eval_strategy="no" \
    --save_strategy="no" \
    --save_total_limit=2 \
    --save_only_model=True \
    --load_best_model_at_end=False \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=2 \
    --eval_on_start=False \
    --report_to="wandb" \
    --bf16=True \
    --num_train_epochs=${epoch} \
    --run_name="lightweight-eos-sft" \
    --eos_sft=${eos_sft}