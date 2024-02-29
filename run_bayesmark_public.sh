# Script to run LLAMBO on all Bayesmark tasks.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# This is the OpenAI LLM Engine
ENGINE="gpt35turbo_20230727"

for dataset in "digits" "wine" "diabetes" "iris" "breast"
do
    for model in "RandomForest" "SVM" "DecisionTree" "MLP_SGD" "AdaBoost"
    do
        python3 exp_bayesmark/run_bayesmark.py --dataset $dataset --model $model --num_seeds 1 --sm_mode discriminative --engine $ENGINE
        sleep 60
    done
done