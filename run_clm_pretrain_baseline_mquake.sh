export CUDA_VISIBLE_DEVICES=1

export WANDB_MODE=online

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
bs=64
per_device_train_batch_size=32
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
warmup_ratio=0.03
max_seq_length=512
epoch=2
# max_steps=1

lr=1e-5

syn_data="mquake_CF3k"
# syn_data="bio_syn_v2"
tunable_params="all"
# tunable_params="top3-mlp"
# tunable_params="midupper3-mlp"

output_dir=models/Llama-3.2-1B-eos-sft-${syn_data}-pretrain-${tunable_params}
# model_name_or_path=${SCRATCH}/base_models/deepseek/hf/deepseek-coder-1.3b-base

python clm_pretrain_baseline_mquake.py \
    --output_dir="${output_dir}" \
    --seed=${seed} \
    --learning_rate=${lr} \
    --lr_scheduler_type=linear \
    --weight_decay=${weight_decay} \
    --per_device_train_batch_size=${per_device_train_batch_size} \
    --per_device_eval_batch_size=${per_device_train_batch_size} \
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
    --load_best_model_at_end=True \
    --logging_strategy="steps" \
    --logging_first_step=True \
    --logging_steps=5 \
    --eval_on_start=True \
    --report_to="wandb" \
    --num_train_epochs=${epoch} \
    --run_name="lightweight-eos-sft" \
    --syn_data=${syn_data} \
    --tunable_params=${tunable_params}
    
    