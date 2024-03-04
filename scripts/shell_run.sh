#!/bin/bash

CFG=configs/scannetpp/scannetppnorm_swinl.json
MODEL=ckpts/nyunormals_swinlarge.pt

MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo "Start script"
python -u scripts/train.py --config-file ${CFG} --model-file ${MODEL} --master-port ${MASTER_PORT} --distributed
