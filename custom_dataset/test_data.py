import pickle

def load_data(data_dir, dataset):
    print("*"*80)
    print(f'Loading data for dataset: {dataset}')
    pickle_fpath = f'{data_dir}/{dataset}.pickle'
    with open(pickle_fpath, 'rb') as f:
        data = pickle.load(f)
    X_train = data['train_x']
    y_train = data['train_y']
    X_test = data['test_x']
    y_test = data['test_y']

    for name in ['train_x', 'train_y', 'test_x', 'test_y']:
        print(f'name: {name}, shape: {data[name].shape}, type: {type(data[name])}')
    for name in ['train_x', 'train_y', 'test_x', 'test_y']:
        print(f'name: {name}, data[{name}][0]:\n {(data[name][0])}')
    print("*"*80)


if __name__ == "__main__":
    # data_dir = 'bayesmark/data'
    # for dataset in ['diabetes', 'breast', 'iris', 'digits', 'wine']:
    #     load_data(data_dir, dataset)

    data_dir = 'custom_dataset/'
    # for dataset in ['Griewank', 'KTablet', 'Rosenbrock', 'CMRR_score', 'Offset_score']:
    for dataset in ['CMRR_score', 'Offset_score']:
        load_data(data_dir, dataset)
