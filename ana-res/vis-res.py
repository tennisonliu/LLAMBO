import json
from pathlib import Path
from rich import print
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 12
ML_MODELS = ["AdaBoost", "DecisionTree", "RandomForest"]
PROVIDERS = ["openai", "ollama"]
STEPS = [5, 10, 20, 30]
# SMS = ["GP", "SMAC", "LLAMBO", "LLAMBO_VANILLA"]
SMS = ["GP", "SMAC", "LLAMBO"]
SMS_LABEL = ["GP", "SMAC", "LLANA"]
# METRICS = ["regret", "rmse", "r2", "nll", "mace", "sharpness", "observed_coverage"]
METRICS = [ "rmse", "r2", "regret", "nll"]
METRICS_LABEL = [r"NRMSE ($\downarrow$)", r"R$^2$ Score ($\uparrow$)", r"Regret ($\downarrow$)", r"LPD ($\downarrow$)"]

RES_ROOT = "./exp_evaluate_sm/results/evaluate_dis_sm"

def get_mean_std_by_sm(s_model, provider, dataset):
    s_model_res = {}
    for ml in ML_MODELS:
        s_model_res[ml] = {}
    for ml in ML_MODELS:
        for step in STEPS:
            with open(f"{RES_ROOT}/{provider}/{dataset}/{ml}/{step}.json", "r") as f:
                res = json.load(f)
                for mts in METRICS:
                    if mts not in s_model_res[ml]:
                        s_model_res[ml][mts] = []
                    s_model_res[ml][mts].append(res[s_model][mts][0])
    s_model_res['all'] = {}
    s_model_res['mean'] = {}
    s_model_res['std'] = {}
    for mts in METRICS:
        s_model_res['all'][mts] = []
        for ml in ML_MODELS:
            s_model_res['all'][mts].append(s_model_res[ml][mts])
        means = np.mean(np.array(s_model_res['all'][mts]), axis=0)
        stds = np.std(np.array(s_model_res['all'][mts]), axis=0)
        s_model_res['mean'][mts] = means
        s_model_res['std'][mts] = stds
    print(f"Surrogate: {s_model}, Provider: {provider}, Dataset: {dataset}")
    for p in ["mean", "std"]:
        print(f"{p}:")
        print(s_model_res[p])
    return s_model_res


if __name__ == "__main__":

    STD_SCALE = 5
    GRID_SIZE = (2, 2)
    METRICS = METRICS[:GRID_SIZE[0]*GRID_SIZE[1]]
    provider = "openai"
    # dataset_name = "CMRR"
    # dataset_name = "Offset"
    for dataset_name in ["CMRR", "Offset"]:
        dataset = f"{dataset_name}_score"
        fig, axs =  plt.subplots(GRID_SIZE[0], GRID_SIZE[1], figsize=(8, 6))  # Set the figure size (width, height) in inches
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                mts = METRICS[i*2+j]
                for sm in SMS:
                    sm_score = get_mean_std_by_sm(sm, provider, dataset)
                    axs[i, j].plot(STEPS, sm_score['mean'][mts], label=SMS_LABEL[SMS.index(sm)])
                    axs[i, j].fill_between(STEPS, sm_score['mean'][mts] - sm_score['std'][mts]/STD_SCALE, sm_score['mean'][mts] + sm_score['std'][mts]/STD_SCALE, alpha=0.2)
                title_name = f"{METRICS_LABEL[METRICS.index(mts)]}"
                axs[i, j].set_title(title_name, fontsize=18)
                axs[i, j].tick_params(axis='both', labelsize=12)
        lines = []
        labels = []
        for ax in axs.flat:
            line, label = ax.get_legend_handles_labels()
            lines.extend(line)
            labels.extend(label)
        fig.legend(lines[:len(SMS)], labels[:len(SMS)], loc='lower center', ncol=4, borderaxespad=0.1,frameon=False,fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        fig.suptitle(f'{dataset_name} dataset', fontsize=20)
        # plt.show()
        save_dir = Path("./ana-res/vis-res/")
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f'{provider}_{dataset}.pdf', dpi=300, bbox_inches='tight')
        plt.close()