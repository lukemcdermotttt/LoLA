distill_checkpoint=""
finetune_checkpoint=""
model_config_path="configs/model/inference_llama3_1_8b_lolcat_sw.yaml"


python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
--finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
--model_config_path "configs/model/inference_llama3_1_8b_lolcat_sw.yaml" \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--task wikitext --num_shots 0 --no_cache --verbose \
--project_name lolcat \
--wandb_entity lmcdermo



#python eval_lm_harness.py \
#--model_type lolcats_ckpt \
#--attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
#--finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
#--model_config_path "configs/model/inference_llama3_1_8b_lolcat_sw.yaml" \
#--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
#--task triviaqa --num_shots 5 --no_cache --verbose \
#--project_name lolcat \
#--wandb_entity lmcdermo


#tasks to try: squad, qasper, truthfulqa, gsm8k, triviaqa, 