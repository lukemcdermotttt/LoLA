distill_checkpoint=""
finetune_checkpoint=""
model_config_path="configs/model/inference_llama3_1_8b_lolcat_sw.yaml"


#python eval_lm_harness.py \
#--model_type lolcats_ckpt \
#--attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
#--finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
##--model_config_path "configs/model/inference_llama3_1_8b_lolcat_sw.yaml" \
#--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
#--task ruler --num_shots 0 --verbose \
#--project_name lolcat \
#--wandb_entity lmcdermo \
#--metadata='{"tokenizer":"meta-llama/Llama-3.1-8B-Instruct"}' #for some reason the base model tokenizer broke so.



python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
--finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
--model_config_path "configs/model/inference_llama3_1_8b_lolcat_sw=64.yaml" \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--task leaderboard_gpqa_diamond --num_shots 0 --verbose \
--project_name lolcat \
--wandb_entity lmcdermo


#tasks to try: squad, qasper, truthfulqa, gsm8k, triviaqa, 