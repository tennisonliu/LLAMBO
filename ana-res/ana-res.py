import pickle
from pathlib import Path

if __name__ == "__main__":
    res_dir = "results"
    for sub_dir in Path(res_dir).iterdir():
        if sub_dir.is_dir():
            for sub_sub_dir in sub_dir.iterdir():
                for file in sub_sub_dir.iterdir():
                    if 'diabetes' in str(file):
                        if file.suffix == ".pickle":
                            with open(file, "rb") as f:
                                data = pickle.load(f)
                                print(file)
                                print(data[0])