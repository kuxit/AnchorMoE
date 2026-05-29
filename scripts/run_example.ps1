$ErrorActionPreference = "Stop"

python .\run.py `
  --task_name classification `
  --is_training 1 `
  --model AnchorMoE `
  --model_id BasicMotions `
  --data UEA `
  --root_path .\data\UEA\Multivariate_ts\BasicMotions `
  --seq_len 100 `
  --d_model 128 `
  --num_groups 4 `
  --patch_len 8 `
  --stride 4 `
  --batch_size 16 `
  --train_epochs 50 `
  --patience 10 `
  --learning_rate 0.001 `
  --optimizer adamw `
  --scheduler cawr

