CONFIG_DIR='/home/ubuntu/linearattention/LoLA/configs/'   # update to your path

# """ Benchmarking the 8b model on the LOLCATS dataset """

# Run the linearized model with the ThunderKittens kernel
CUDA_VISIBLE_DEVICES=0 python -Wignore demo_lolcats_hf.py \
    --model_config_path ${CONFIG_DIR}/model/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml \
    --distill_config_path ${CONFIG_DIR}/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-2.yaml \
    --finetune_config_path ${CONFIG_DIR}/experiment/finetune_lora_qkvo_alpaca_clean.yaml \
    --attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
    --finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
    --num_generations 1 \
    --use_cuda_kernels 1 \
    --benchmark \
    --max_new_tokens 1

# Run the linearized model *without* the ThunderKittens kernel
CUDA_VISIBLE_DEVICES=0 python -Wignore demo_lolcats_hf.py \
    --model_config_path ${CONFIG_DIR}/model/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml \
    --distill_config_path ${CONFIG_DIR}/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-2.yaml \
    --finetune_config_path ${CONFIG_DIR}/experiment/finetune_lora_qkvo_alpaca_clean.yaml \
    --attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
    --finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
    --num_generations 1 \
    --use_cuda_kernels 0 \
    --benchmark \
    --max_new_tokens 1

# Run the base Llama model with Transformers SDPA attention
CUDA_VISIBLE_DEVICES=0 python -Wignore demo_lolcats_hf.py \
    --model_config_path ${CONFIG_DIR}/model/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml \
    --distill_config_path ${CONFIG_DIR}/experiment/distill_alpaca_clean_xent0_mse1000_lr1e-2.yaml \
    --finetune_config_path ${CONFIG_DIR}/experiment/finetune_lora_qkvo_alpaca_clean.yaml \
    --attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
    --finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
    --num_generations 1 \
    --use_attention \
    --benchmark \
    --max_new_tokens 1
