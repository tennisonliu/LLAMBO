import pickle
from pathlib import Path
import pandas as pd
from rich import print
import matplotlib.pyplot as plt


ALL_SCORE_MODELS = [
    'ada',
    'DT',
    'MLP-sgd',
    'RF',
    'SVM'
]

SCORE_MODEL_MAPPING = {
    'ada': 'AdaBoost',
    'DT': 'DecisionTree',
    'MLP-sgd': 'MLP_SGD',
    'RF': 'RandomForest',
    'SVM': 'SVM'
}

ALL_BO_MODELS = [
    'DNGO',
    'GP',
    'GP_DKL',
    'Optuna',
    'Random',
    'SKOPT',
    'TPE',
    'Turbo',
]

ALL_DATASETS = [
    'CMRR',
    'Offset'
]

ALL_SCORE_METRICS = [
    'score',
    'generalization_score',
    'mean_absolute_error',
    'mean_squared_error',
    'obj_loss'
]

def ana_baseline():
    res_dir = "results"
    for sub_dir in Path(res_dir).iterdir():
        # sub_dir : metric
        if sub_dir.is_dir():
            # sub_sub_dir: BO models
            for sub_sub_dir in sub_dir.iterdir():
                if sub_sub_dir.name.replace('_(Random_init)','') not in ALL_BO_MODELS:
                    print(sub_sub_dir.name)
                for file in sub_sub_dir.iterdir():
                    if 'metrics' in str(file):
                        if file.suffix == ".pickle":
                            with open(file, "rb") as f:
                                data = pickle.load(f)
                                # print(file)
                                # print(data[0])

# def info_baseline(data):
#     data_numeric = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
#     mean_all = data_numeric.mean()
#     print(mean_all, '\n')

# def info_llm(data):
#     data_numeric = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
#     mean_all = data_numeric.mean()
#     print(mean_all, '\n')

def info_data(data, show=False):
    data_numeric = data.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    # show_set = set(data_numeric.columns).intersection(set(ALL_SCORE_METRICS))
    # data_numeric = data_numeric[list(show_set)]
    mean_all = data_numeric.mean()
    min_all = data_numeric.min()
    if show:
        print(mean_all, '\n')
        print('Min:')
        print(min_all, '\n')
        print('-'*5)
    return mean_all


def draw_data(data, title, y='generalization_score'):
    data.plot(y=y)
    plt.title(title)
    plt.show()

def get_baseline_model(score_model, bo_model, dataset):
    path = Path("results") / f'metric_mse_data_{dataset}_score_model_{score_model}/{bo_model}_(Random_init)/metrics.pickle'
    title = f"[Baseline] Score model: {score_model}, BO model: {bo_model}, Data: {dataset}"
    if not path.exists():
        return
    with open(str(path), "rb") as f:
        data = pickle.load(f)
    print(title)
    data = data[0]
    # draw_data(data, title=title, y="mean_squared_error")
    try:
        return info_data(data, show=False)
    except Exception as e:
        raise(e)


def get_llm_model(provider, llm, dataset, score_model):
    path = Path("exp_custom") / f'{provider}' / f'{llm}' / 'results_discriminative' / f'{dataset}_score/{SCORE_MODEL_MAPPING[score_model]}/0.csv'
    title = f"[LLM]: {provider}: {llm}, Data: {dataset}, Score model: {score_model}"
    try:
        df = pd.read_csv(str(path))
    except:
        print(path)
        print(title)
        raise('Error loading file')
    print(title)
    print(f"record shape: {df.shape}")
    info_data(df, show=True)
    # draw_data(df, title=title)
    return df

if __name__ == "__main__":
    # ana_baseline()
    for score_model in ALL_SCORE_MODELS:
        for dataset in ALL_DATASETS:
            print('*'*20, f'{score_model}, {dataset}', '*'*20)
            all_bo_means = []
            for bo_model in ALL_BO_MODELS:
                mean_data = get_baseline_model(score_model, bo_model, dataset)
                all_bo_means.append(mean_data)
            bo_mean = pd.concat(all_bo_means, axis=1).T
            print('*'*20, score_model, dataset, 'bo_mean', '*'*20, '\n')
            print(f'[Baseline mean]\n {bo_mean.mean()}')
            print(f'[Baseline min]\n {bo_mean.min()} \n')
            llm_data = get_llm_model('openai', 'gpt-3.5-turbo', dataset, score_model)
            llm_data = get_llm_model('ollama', 'mistral', dataset, score_model)
            print('*'*20, score_model, '*'*20, '\n')