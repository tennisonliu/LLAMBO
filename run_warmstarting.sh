#!/bin/bash
trap "kill -- -$BASHPID" EXIT

# Creating initial points
for config in "random", "sobol", "lhs", "Partial_Context", "No_Context", "Full_Context"
do
    python3 exp_warmstarting/init_warmstart.py --config $config
    sleep 60
done

# It assumes  "random", "sobol", "lhs", "Partial_Context", "No_Context", "Full_Context"
python3 exp_warmstarting/run_warmstart.py