export CUDA_VISIBLE_DEVICES=3

gpu_count=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
bs=1
per_device_train_batch_size=1
grad_acc=$((bs / gpu_count / per_device_train_batch_size))

weight_decay=0.1
max_grad_norm=1.0

seed=42
max_seq_length=1024
epoch=4
# max_steps=1

lr=1e-5

# input_format=2hop

# second-1hop

echo grad_acc: ${grad_acc}
# tunable_params="all"
# base_model_name="Llama-3.2-1B-common-country-eos-sft"

# tunable_params="midupper3-mlp"

base_model_name="Qwen2.5-1.5B-eos-sft-template-format-curated-v1-lr2e-6-sample-10"

# date_data="all_propagation_ood"
# date_data="all_propagation_ood_w_ood_country"

date_data=test_id
text_data="augmented_texts"

# tunable_params="midupper-mlp"
tunable_params="all"

# for tunable_params in "midupper-mlp" # "midupper3-mlp" # "all" 
# do 
for example_idx in {8..499} # {0..349} # {0..499}
do

echo "Test data: ${date_data}"
echo "Example idx: ${example_idx}"

python clm_baseline_syn_story_qwen_meta-aug.py \
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
    --run_name="propagator-clm-baseline" \
    --example_idx=${example_idx} \
    --report_to="none" \
    --spec_question=False \
    --date_data=${date_data} \
    --text_data=${text_data} \
    --tunable_params=${tunable_params} \
    --base_model_name=${base_model_name} 

done
# done

# date_data="test_ood"

# for tunable_params in "all" "midupper3-mlp"
# do 
# for example_idx in {0..99} #..99} #  47 48 49 51 52 53 54 59 71 72 {75..99} # {0..99} # {0..999}
# do
# echo "Test data: ${date_data}"
# echo "Example idx: ${example_idx}"

# # accelerate launch --config_file="fsdp_config.yaml" \
#     # --main_process_port 29700 \
# python clm_baseline_syn_story.py \
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
#     --run_name="propagator-clm-baseline" \
#     --example_idx=${example_idx} \
#     --report_to="none" \
#     --spec_question=False \
#     --date_data=${date_data} \
#     --text_data=${text_data} \
#     --tunable_params=${tunable_params} \
#     --base_model_name=${base_model_name} 

# done
# done
# for date_data in "test_ood" "test_ood-entity" "test_ood-relation"
# do
# for example_idx in {0..349}
# do

# echo "Test data: ${date_data}"
# echo "Example idx: ${example_idx}"

# python clm_baseline_syn_story_qwen.py \
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
#     --run_name="propagator-clm-baseline" \
#     --example_idx=${example_idx} \
#     --report_to="none" \
#     --spec_question=False \
#     --date_data=${date_data} \
#     --text_data=${text_data} \
#     --tunable_params=${tunable_params} \
#     --base_model_name=${base_model_name} 

# done
# done