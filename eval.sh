# python run.py --mode eval --config evoenc/config/evoenc_s1.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s1/ckpt.EEPolicy.epoch0.pth RESULTS_DIR data/checkpoints/evoenc_s1/evals
# python run.py --mode eval --config evoenc/config/evoenc_s2.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s1/ckpt.EEPolicy.epoch0.pth RESULTS_DIR data/checkpoints/evoenc_s1/evals


# python run.py --mode eval --config evoenc/config/evoenc_s1.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s2/ckpt.EEPolicy.epoch0.pth RESULTS_DIR data/checkpoints/evoenc_s2/evals
# python run.py --mode eval --config evoenc/config/evoenc_s2.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s2/ckpt.EEPolicy.epoch0.pth  RESULTS_DIR data/checkpoints/evoenc_s2/evals

# python run.py --mode eval --config evoenc/config/evoenc_s3.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s1/ckpt.EEPolicy.epoch0.pth TORCH_GPU_ID 1 RESULTS_DIR data/checkpoints/evoenc_s1/evals
# python run.py --mode eval --config evoenc/config/evoenc_s3.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s2/ckpt.EEPolicy.epoch0.pth TORCH_GPU_ID 1  RESULTS_DIR data/checkpoints/evoenc_s2/evals

python run.py --mode eval --config evoenc/config/evoenc_s3.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s3/ckpt.EEPolicy.epoch2.pth TORCH_GPU_ID 1 RESULTS_DIR data/checkpoints/evoenc_s3/evals
python run.py --mode eval --config evoenc/config/evoenc_s1.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s3/ckpt.EEPolicy.epoch2.pth RESULTS_DIR data/checkpoints/evoenc_s3/evals
python run.py --mode eval --config evoenc/config/evoenc_s2.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_s3/ckpt.EEPolicy.epoch2.pth  RESULTS_DIR data/checkpoints/evoenc_s3/evals