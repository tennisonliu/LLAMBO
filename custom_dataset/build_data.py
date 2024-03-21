import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split

def load_weights_from_file(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:  # Ensure there are exactly two elements (key and value)
                key = parts[0]
                try:
                    value = float(parts[1])
                    result_dict[key] = value
                except ValueError:
                    print(f"Error converting value for {key} to float.")
            else:
                print(f"Skipping invalid line: {line.strip()}")
    return result_dict

def load_perf_from_file(filename):
    result_dict = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split('=')
            if len(parts) == 2:  # Ensure there are exactly two elements (key and value)
                key = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    result_dict[key] = value
                except ValueError:
                    print(f"Error converting value for {key} to float.")
            else:
                print(f"Skipping invalid line: {line.strip()}")
    return result_dict

# Example usage

def get_nets_weight_perf(data_json_path):
    # data_json_path = 'custom_dataset/cmrr_data.json'
    cmrr_json = json.load(open(data_json_path))
    all_cnets = cmrr_json['all_cnets']
    # print(f'all_cnets: {len(all_cnets)}')
    data_dir = cmrr_json['dir_prefix']
    objs = cmrr_json['obj']
    gls = cmrr_json['gl']
    save_dir = cmrr_json['save_dir']
    sub_dirs = [sub_dir for sub_dir in Path(data_dir).iterdir() if sub_dir.is_dir()]
    all_cnets_data = []
    for sub_dir in sub_dirs:
        weight_txt = sub_dir / 'weight.txt'
        perf_txt = sub_dir / 'performance.txt'
        if not weight_txt.exists():
            raise(f"Weight file {weight_txt} does not exist.")
        data_dict = load_weights_from_file(weight_txt)
        cnet_dict = {}
        for key in data_dict.keys():
            if key in all_cnets:
                cnet_dict[key] = data_dict[key]
        perf_dict = load_perf_from_file(perf_txt)
        cnet_dict.update(perf_dict)
        all_cnets_data.append(cnet_dict)
    df = pd.DataFrame(all_cnets_data)
    for obj in objs:
        obj_base_val = cmrr_json['obj_val'][objs.index(obj)]
        gl = gls[objs.index(obj)]
        gl_idx = -1 if gl == 'max' else 1
        df[f'{obj}_score'] = df[obj].apply(lambda x: gl_idx * (x - obj_base_val) / obj_base_val)

    x = df.iloc[:, :len(all_cnets)]
    for obj in objs:
        score_loc_id = len(all_cnets) + len(objs) + objs.index(obj)
        y = df.iloc[:, score_loc_id]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)
            # Converting to numpy arrays
        data = {
            'train_x': train_x.to_numpy(),
            'train_y': train_y.to_numpy(),
            'test_x': test_x.to_numpy(),
            'test_y': test_y.to_numpy()
        }
        print(f'train_x[0]: {data["train_x"][0]}')
        print(f'train_y[0]: {data["train_y"][0]}')
        pickle_fpath = f'{save_dir}/{obj}_score.pickle'
        with open(pickle_fpath, 'wb') as f:
            pickle.dump(data, f)
            print(f'Saved data to {pickle_fpath}')

    return df


def get_nets(arg):
    pass




if __name__ == '__main__':
    df = get_nets_weight_perf('custom_dataset/cmrr_data.json')
    print(df)

    # Assuming df is your DataFrame
    # Let's say the first four columns are features and the fifth column is the target
    # all_cnets 14
    # cmrr_score_id = 16
    # offset_score_id = 17
    # X = df.iloc[:, :14]  # Features: First four columns
    # y = df.iloc[:, 16]   # Target: Fifth column

    # # Splitting the data into training and testing sets (80% train, 20% test)
    # train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Converting to numpy arrays
    # train_x = train_x.to_numpy()
    # test_x = test_x.to_numpy()
    # train_y = train_y.to_numpy()
    # test_y = test_y.to_numpy()



