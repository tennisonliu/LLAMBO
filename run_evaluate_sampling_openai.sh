# Script to evaluate candidate point sampler.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

provider="openai"
ENGINE="gpt-3.5-turbo"

for dataset in "CMRR_score" "Offset_score"
do
    for model in "RandomForest" "SVM" "AdaBoost"
    do
        for num_observed in 5 10 20 30
        do
            python3 exp_evaluate_sampling/evaluate_sampling.py --dataset $dataset --model $model --num_observed $num_observed --num_seeds 1 --engine $ENGINE --provider $provider
        done
    done
done