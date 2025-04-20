python distill_llama.py --model_config distill_llama3_2_1b_lk_smd_wsw64_fd64_w01_sparse \
--distill_config no_distill_alpaca_clean \
--finetune_config finetune_lora_qkvo_alpaca_clean \
--eval_config eval_alpaca_clean \
--lk_zero_init \
--verbose \
--seed 0 \
--replicate 0 \
--project_name lolcat \
--wandb_entity lmcdermo \
--load_distill_checkpoint ./checkpoints/distill_llama3_2_1b_lk_smd_wsw64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_2_1b_lk_smd_wsw64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=0-lzi=1_distill.pt

#distill_alpaca_clean_xent0_mse1000_lr1e-2