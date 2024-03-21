# Script to run LLAMBO on all Bayesmark tasks.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# This is the OpenAI LLM Engine
provider="openai"
ENGINE="gpt-3.5-turbo"

for dataset in "digits" "wine" "diabetes" "iris" "breast"
do
    for model in "RandomForest" "SVM" "DecisionTree" "MLP_SGD" "AdaBoost"
    # for model in "SVM" "DecisionTree" "MLP_SGD" "AdaBoost"
    # for model in "DecisionTree" "MLP_SGD" "AdaBoost"
    # for model in "MLP_SGD" "AdaBoost"
    # for model in "AdaBoost"
    do
        python3 exp_bayesmark/run_bayesmark.py --dataset $dataset --model $model --num_seeds 1 --sm_mode discriminative --engine $ENGINE --provider $provider
        sleep 60
    done
done