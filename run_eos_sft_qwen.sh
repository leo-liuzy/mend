export CUDA_VISIBLE_DEVICES=2

export WANDB_MODE=online

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")


eos_sft=True
do_lora=False

if [ "$eos_sft" = True ]; then
    # this is for eos-sft
    bs=128
    per_device_train_batch_size=16
    lr=5e-6 # this is for eos-sft
    output_dir=models/Llama3-8B-eos-sft
else
    # this is for template-sft
    bs=10 
    per_device_train_batch_size=2
    lr=2e-6 
fi


grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
warmup_ratio=0.03
max_seq_length=512
epoch=2

if [ "$do_lora" = True ]; then
    # lora
    
    if [ "$eos_sft" = True ]; then
        # this is for eos-sft
        lr=1e-4
        output_dir=models/Qwen2.5-1.5B-eos-sft-lr${lr}
    else
        # this is for template-sft
        lr=2e-5
        output_dir=models/Qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr${lr}-sample-10
    fi
    output_dir=${output_dir}-lora
fi

# output_dir=models/Qwen2.5-1.5B-eos-sft-lora # -template-format-curated-v1-lr${lr}-sample-10
# output_dir=models/Qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr${lr}-sample-10
# model_name_or_path=${SCRATCH}/base_models/deepseek/hf/deepseek-coder-1.3b-base

# accelerate launch --config_file="fsdp_config.yaml" \
#     --main_process_port 29601 \
python lightweight_sft_qwen.py \
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
    --load_best_model_at_end=False \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=2 \
    --eval_on_start=False \
    --report_to="wandb" \
    --num_train_epochs=${epoch} \
    --run_name="lightweight-eos-sft"\
    --is_peft=${do_lora} \
    --eos_sft=${eos_sft} \
    # --bf16=True \