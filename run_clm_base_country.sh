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
epoch=4

# second-1hop


# tunable_params="all"
# base_model_name="Llama-3.2-1B-common-country-eos-sft"

# tunable_params="midupper3-mlp"
tunable_params="all"
# base_model_name="Llama-3.2-1B-common-country-eos-sft-country_syn-pretrain-all"
base_model_name="Llama-3.2-1B-common-country-eos-sft"

# date_data="all_propagation_ood"
# date_data="all_propagation_ood_w_ood_country"

for date_data in all_propagation_ood_v2 # all_propagation_ood_w_ood_country_v2 all_propagation_ood_w_ood_country # all_propagation_ood_w_ood_country #  
do 
    for example_idx in {0..99} #  47 48 49 51 52 53 54 59 71 72 {75..99} # {0..99} # {0..999}
    do

    for text_data in "text" 
    do

    echo "Example idx: ${example_idx}"

    # accelerate launch --config_file="fsdp_config.yaml" \
        # --main_process_port 29700 \
    python clm_baseline_country_ood.py \
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
    done
done