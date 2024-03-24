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
    # for name in ['train_x', 'train_y', 'test_x', 'test_y']:
    #     print(f'name: {name}, data[{name}][0]:\n {(data[name][0])}')

    for name in ['train_y', 'test_y']:
        print(f'max {name}: {max(data[name])}, min {name}: {min(data[name])}')
    # global_perf = {
    #     'train_x_max': [],
    #     'train_x_min': [],
    #     'test_x_max': [],
    #     'test_x_min': []
    # }
    # for name in ['train_x', 'test_x']:
    #     for i in range(data[name].shape[0]):
    #         print(f'{name}[{i}] shape {data[name][i].shape}')
    #         print(f'max {name}: {max(data[name][i])}, min {name}: {min(data[name][i])}')
    print("*"*80)


if __name__ == "__main__":
    # data_dir = 'bayesmark/data'
    # for dataset in ['diabetes', 'breast', 'iris', 'digits', 'wine']:
    #     load_data(data_dir, dataset)
    # for dataset in ['Griewank', 'KTablet', 'Rosenbrock', 'CMRR_score', 'Offset_score']:

    data_dir = 'custom_dataset/'
    for dataset in ['CMRR_score', 'Offset_score']:
        load_data(data_dir, dataset)


    # data_dir = 'bayesmark/data'
    # for dataset in ['wine', 'breast', 'digits', 'diabetes']:
    #     load_data(data_dir, dataset)
