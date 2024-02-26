python run.py --run-type eval --exp-config evoenc/config/evoenc_p0.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p0 RESULTS_DIR data/checkpoints/evoenc_p0/evals
python run.py --run-type eval --exp-config evoenc/config/evoenc_p0.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p1 RESULTS_DIR data/checkpoints/evoenc_p1/evals
python run.py --run-type eval --exp-config evoenc/config/evoenc_p0.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p2 RESULTS_DIR data/checkpoints/evoenc_p2/evals

python run.py --run-type eval --exp-config evoenc/config/evoenc_p1.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p0 RESULTS_DIR data/checkpoints/evoenc_p0/evals
python run.py --run-type eval --exp-config evoenc/config/evoenc_p1.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p1 RESULTS_DIR data/checkpoints/evoenc_p1/evals
python run.py --run-type eval --exp-config evoenc/config/evoenc_p1.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p2 RESULTS_DIR data/checkpoints/evoenc_p2/evals

python run.py --run-type eval --exp-config evoenc/config/evoenc_p2.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p0 RESULTS_DIR data/checkpoints/evoenc_p0/evals
python run.py --run-type eval --exp-config evoenc/config/evoenc_p2.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p1 RESULTS_DIR data/checkpoints/evoenc_p1/evals
python run.py --run-type eval --exp-config evoenc/config/evoenc_p2.yaml EVAL_CKPT_PATH_DIR data/checkpoints/evoenc_p2 RESULTS_DIR data/checkpoints/evoenc_p2/evals