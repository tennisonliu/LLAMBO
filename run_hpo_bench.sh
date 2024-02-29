# Script to run LLAMBO on all HPOBench tasks.


#!/bin/bash
trap "kill -- -$BASHPID" EXIT

ENGINE="gpt35turbo_20230727"

for dataset in "australian" "blood_transfusion" "car" "credit_g" "kc1" "phoneme" "segment" "vehicle"
do
    for model in "rf" "xgb" "nn"
    do
        echo "dataset: $dataset, model: $model"
        python3 exp_hpo_bench/run_hpo_bench.py --dataset $dataset --model $model --seed 0 --num_seeds 1 --engine $ENGINE --sm_mode discriminative
    done
done