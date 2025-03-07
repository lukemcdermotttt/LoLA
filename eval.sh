distill_checkpoint="./checkpoints/llama3_1b_prune_d64_halfspace/llama3_1b_D64_dense_halfspace-se=0-re=0-lzi=1_distill.pt"
finetune_checkpoint="./checkpoints/llama3_1b_prune_d64_SW32_halfspace/llama3_1b_D64_halfspace_dense_noSWdist_SW32_finetune-se=0-re=0-lzi=1-se=0-re=0_ft.pt"
model_config_path="configs/model/llama3_1b_d64_SWHybrid_halfspace.yaml"

"""
python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path  $distill_checkpoint \
--finetune_checkpoint_path  $finetune_checkpoint \
--model_config_path $model_config_path \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--project_name lolcat \
--wandb_entity lmcdermo \
--task piqa \
--num_shots 0  \
--no_cache \
--verbose

python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path  $distill_checkpoint \
--finetune_checkpoint_path  $finetune_checkpoint \
--model_config_path $model_config_path \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--project_name lolcat \
--wandb_entity lmcdermo \
--task arc_easy \
--num_shots 0  \
--no_cache \
--verbose
"""

python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path  $distill_checkpoint \
--finetune_checkpoint_path  $finetune_checkpoint \
--model_config_path $model_config_path \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--project_name lolcat \
--wandb_entity lmcdermo \
--task arc_challenge \
--num_shots 0  \
--no_cache \
--verbose

"""

python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path  $distill_checkpoint \
--finetune_checkpoint_path  $finetune_checkpoint \
--model_config_path $model_config_path \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--project_name lolcat \
--wandb_entity lmcdermo \
--task hellaswag \
--num_shots 0  \
--no_cache \
--verbose


python eval_lm_harness.py \
--model_type lolcats_ckpt \
--attn_mlp_checkpoint_path  $distill_checkpoint \
--finetune_checkpoint_path  $finetune_checkpoint \
--model_config_path $model_config_path \
--finetune_config_path 'configs/experiment/finetune_lora_qkvo_alpaca_clean.yaml' \
--project_name lolcat \
--wandb_entity lmcdermo \
--task winogrande \
--num_shots 0  \
--no_cache \
--verbose
"""