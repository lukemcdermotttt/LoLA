distill_checkpoint=""
finetune_checkpoint=""
model_config_path="configs/model/inference_llama3_1_8b_lolcat_sw.yaml"

python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
--finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
--model_config_path "configs/model/inference_llama3_1_8b_baseline.yaml" \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--task niah_single_1 --num_shots 0 --verbose \
--project_name lolcat \
--wandb_entity lmcdermo \
--metadata='{"max_seq_lengths":[512],"tokenizer":"meta-llama/Llama-3.1-8B-Instruct"}' #for some reason the base model tokenizer broke so.

#python eval_lm_harness.py \
#--model_type lolcats_ckpt \
#--attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
#--finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
#--model_config_path "configs/model/inference_llama3_1_8b_lola.yaml" \
#--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
#--task lambada_openai --num_shots 0 --verbose \
#--project_name lolcat \
#--wandb_entity lmcdermo

#Llama 1B NO SLIDING WINDOW
#python eval_lm_harness.py \
#--model_type lolcats_ckpt \
#--attn_mlp_checkpoint_path '/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64/llama3_1b_D64_dense-se=0-re=0-lzi=1_distill.pt' \
#--finetune_checkpoint_path '/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64/llama3_1b_D64_dense_finetune-se=0-re=0-lzi=1-se=0-re=0_ft.pt' \
#--model_config_path "configs/model/inference_llama3_2_1b_global=0_local=0.yaml" \
#--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
#--task arc_easy --num_shots 0 --verbose \
#--project_name lolcat \
#--wandb_entity lmcdermo

#Llama 1B w/ sliding window
#python eval_lm_harness.py \
#--model_type lolcats_ckpt \
#--attn_mlp_checkpoint_path '/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64/llama3_1b_D64_dense-se=0-re=0-lzi=1_distill.pt' \
#--finetune_checkpoint_path '/home/archy2/luke/lolcats/checkpoints/llama3_1b_prune_d64_SW/llama3_1b_D64_dense_SW_finetune-se=0-re=0-lzi=1-se=0-re=0_ft.pt' \
#--model_config_path "configs/model/inference_llama3_2_1b_global=0_local=64.yaml" \
#--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
#--task arc_challenge --num_shots 0 --verbose \
#--project_name lolcat \
#--wandb_entity lmcdermo

#tasks to try: squad, qasper, truthfulqa, gsm8k, triviaqa, 