python distill_llama.py --model_config distill_llama3_2_1b_delta \
--distill_config no_distill_alpaca_clean \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose \
--seed 0 \
--replicate 0 \
--project_name lolcat \
--wandb_entity lmcdermo

#distill_alpaca_clean_xent0_mse1000_lr1e-2