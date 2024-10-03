accelerate launch --config_file "configs/fsdp_config_allparams.yaml" train.py \
--seed 100 \
--model_name_or_path "/home/bbadger/Desktop/llama-3.1-8b-instruct" \
--dataset_path "open-phi/textbooks" \
--add_special_tokens False \
--append_concat_token False \
--max_seq_len 256 \
--num_train_epochs 8 \
--logging_steps 50 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "epoch" \
--save_strategy "epoch" \
--bf16 False \
--packing False \
--learning_rate 2e-5 \
--lr_scheduler_type "linear" \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "/home/bbadger/experiments/full_llama3.1_8b" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_checkpointing True \
--dataset_text_field "content" \
--use_flash_attn False \
--use_peft_lora False \
--report_to "none"