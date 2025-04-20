distill_checkpoint="./checkpoints/distill_llama3_2_1b_lk_smd_wsw64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_2_1b_lk_smd_wsw64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=0-lzi=1_distill.pt" #"/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64/llama3_1b_D64_dense-se=0-re=0-lzi=1_distill.pt"
finetune_checkpoint="./checkpoints/distill_llama3_2_1b_lk_smd_wsw64_fd64_w01/dl-d=dacxmldl21lswfwflqac000_lzi=1_distill1d-m=distill_llama3_2_1b_lk_smd_wsw64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=0-lzi=1-se=0-re=0_ft.pt"  #"/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64_SW/llama3_1b_D64_dense_SW_finetune-se=0-re=0-lzi=1-se=0-re=0_ft.pt" #"/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64/llama3_1b_D64_dense_finetune-se=0-re=0-lzi=1-se=0-re=0_ft.pt"
model_config_path="configs/model/inference_llama3_2_1b_prefill.yaml" #"configs/model/inference_llama3_2_1b_prefill.yaml" #"configs/model/inference_llama3_2_1b_global=0_local=64.yaml" #"configs/model/inference_llama3_2_1b_global=0_local=0.yaml"


#1B SW DISTILL CHECKPOINT: ./checkpoints/distill_llama3_2_1b_lk_smd_wsw64_fd64_w01/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_2_1b_lk_smd_wsw64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=0-lzi=1_distill.pt
#1B SW FT CHECKPOINT: ./checkpoints/distill_llama3_2_1b_lk_smd_wsw64_fd64_w01/dl-d=dacxmldl21lswfwflqac000_lzi=1_distill1d-m=distill_llama3_2_1b_lk_smd_wsw64_fd64_w01-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=0-lzi=1-se=0-re=0_ft.pt

#1B NoSW FT: ./checkpoints/distill_llama3_2_1b_lk_smd_fd64/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_2_1b_lk_smd_fd64-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=0-lzi=1_distill.pt
#1B NoSW FT: ./checkpoints/distill_llama3_2_1b_lk_smd_fd64/dl-d=distill_alpaca_clean_xent0_mse1000_lr1e-2-m=distill_llama3_2_1b_lk_smd_fd64-f=finetune_lora_qkvo_alpaca_clean-s=0-se=0-re=0-lzi=1-bs=1-gas=8-nte=2-ms=-1-se=0-re=0_ft.pt

#Llama 1B NO SLIDING WINDOW
python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path $distill_checkpoint \
--finetune_checkpoint_path $finetune_checkpoint \
--model_config_path $model_config_path \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--task lambada_openai --num_shots 0 --verbose --no_cache \
--project_name lolcat \
--wandb_entity lmcdermo


#Llama 1B w/ sliding window
#python eval_lm_harness.py \
#--model_type lolcats_ckpt \
#--attn_mlp_checkpoint_path '/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64/llama3_1b_D64_dense-se=0-re=0-lzi=1_distill.pt' \
#--finetune_checkpoint_path '/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64_SW/llama3_1b_D64_dense_SW_finetune-se=0-re=0-lzi=1-se=0-re=0_ft.pt' \
#--model_config_path "configs/model/inference_llama3_2_1b_global=0_local=64.yaml" \
#--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
#--task arc_challenge --num_shots 0 --verbose --no_cache \
#--project_name lolcat \
#--wandb_entity lmcdermo


#python eval_lm_harness.py \
#--model_type lolcats_ckpt \
#--attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
#--finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
#--model_config_path "configs/model/inference_llama3_1_8b_lolcat_sw.yaml" \
#--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
#--task triviaqa --num_shots 5 --no_cache --verbose --no_cache \
#--project_name lolcat \
#--wandb_entity lmcdermo


#tasks to try: squad, qasper, truthfulqa, gsm8k, triviaqa, 