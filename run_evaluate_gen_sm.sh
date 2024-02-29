# Script to evaluate generative surrogate model.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

for dataset in "breast" "wine" "digits" "diabetes"
do
    for model in "RandomForest" "DecisionTree" "SVM"
    do
        for num_observed in 5 10 20 30
        do
            python3 exp_evaluate_sm/evaluate_gen_sm.py --dataset $dataset --model $model --num_observed $num_observed --num_seeds 1
        done
    done
done