# Script to run ablation study on prompt design.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

ENGINE="gpt35turbo_20230727"

for dataset in "breast"
do
    for model in "RandomForest"
    do
        for ablation_type in "no_context" "partial_context" "full_context"
        do
            python3 exp_prompt_ablation/run_ablation.py --dataset $dataset --model $model --num_seeds 1 --sm_mode discriminative --engine $ENGINE --ablation_type $ablation_type --shuffle_features False
        done
    done
done