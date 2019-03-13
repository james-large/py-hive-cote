import numpy as np
import pandas as pd

def load_csv(file_path):
    with open(file_path) as f:
        for line in f:
            if line.strip():
                if "@data" in line.lower():
                    data_started = True
                    break

        dataset = np.loadtxt(f,delimiter=",")
        X = dataset[:,0:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
    return X,Y

def load_to_pandas(file_path):
    with open(file_path) as f:
        for line in f:
            if line.strip():
                if "@data" in line.lower():
                    data_started = True
                    break
        df = pd.read_csv(f, delimiter=',', header=None)
        Y = df.pop(df.shape[1] - 1)
        X = pd.DataFrame([[row] for _, row in df.iterrows()])
    # transform into nested pandas dataframe
    return X,Y

