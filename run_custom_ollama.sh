# Script to run LLAMBO on all Bayesmark tasks.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# This is the OpenAI LLM Engine
provider="ollama"
ENGINE="mistral"

# for dataset in "CMRR_score"
for dataset in "Offset_score"
do
    for model in "RandomForest" "SVM" "DecisionTree" "MLP_SGD" "AdaBoost"
    # for model in "SVM" "DecisionTree" "MLP_SGD" "AdaBoost"
    # for model in "DecisionTree" "MLP_SGD" "AdaBoost"
    # for model in "MLP_SGD" "AdaBoost"
    # for model in "AdaBoost"
    do
        python3 exp_custom/run_custom.py --dataset $dataset --model $model --num_seeds 1 --sm_mode discriminative --engine $ENGINE --provider $provider
        sleep 10
    done
done
