# Script to evaluate discriminative surrogate model.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

provider="openai"
ENGINE="gpt-3.5-turbo"

for dataset in "CMRR_score" "Offset_score"
do
    # for model in "RandomForest" "SVM" "AdaBoost"
    for model in "DecisionTree" "MLP_SGD"
    do
        for num_observed in 5 10 20 30
        do
            python3 exp_evaluate_sm/evaluate_dis_sm.py --dataset $dataset --model $model --num_observed $num_observed --num_seeds 1 --engine $ENGINE --provider $provider
        done
    done
done