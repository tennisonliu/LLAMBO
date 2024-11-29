# Script to run LLAMBO on all Bayesmark tasks.

#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# This is the OpenAI LLM Engine
# gpt35turbo_20230727 is no longer available it seems
ENGINE="gpt-3.5-turbo-0125"

for method in "RandomForest" "SVM" "AdaBoost"
do
    for gamma in "1" "0.75" "0.5"
    do
        for budget in "5" "10" "15" "20" "25"
        do
          echo $method $gamma $budget
          nohup python3 exp_bayesmark/run_bayesmark.py --dataset digits --model $method --num_seeds 5 --sm_mode discriminative --engine $ENGINE --cost_aware true --mab_exp_smoothing_gamma $gamma --budget $budget &
          sleep 3
          nohup python3 exp_bayesmark/run_bayesmark.py --dataset wine --model $method --num_seeds 5 --sm_mode discriminative --engine $ENGINE --cost_aware true --mab_exp_smoothing_gamma $gamma --budget $budget &
          sleep 3
          nohup python3 exp_bayesmark/run_bayesmark.py --dataset diabetes --model $method --num_seeds 5 --sm_mode discriminative --engine $ENGINE --cost_aware true --mab_exp_smoothing_gamma $gamma --budget $budget &
          sleep 3
          nohup python3 exp_bayesmark/run_bayesmark.py --dataset iris --model $method --num_seeds 5 --sm_mode discriminative --engine $ENGINE --cost_aware true --mab_exp_smoothing_gamma $gamma --budget $budget &
          sleep 3
          nohup python3 exp_bayesmark/run_bayesmark.py --dataset breast --model $method --num_seeds 5 --sm_mode discriminative --engine $ENGINE --cost_aware true --mab_exp_smoothing_gamma $gamma --budget $budget &
          wait
          echo "all done"""
        done
    done
done

