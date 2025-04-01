import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from conformalization import cp, coverage_and_length

if __name__ == "__main__":
    path = "../results/uncertainty/conformal_results_start1_end99.npz"
    cp_results = np.load(path, allow_pickle=True) 
    print(f"keys: {cp_results.files}")

    cov_dict = {}
    len_dict = {}
    empty_dict = {}
    for type in ['LABEL', 'APS', 'MCP_LABEL', 'MCP_APS']:
        for evaluate in ['coverage', 'length', 'empty']:
            
            if evaluate == 'coverage':
                cov_dict[type] = cp_results[f"{type}_{evaluate}"]
            elif evaluate == 'length':
                len_dict[type] = cp_results[f"{type}_{evaluate}"]
            elif evaluate == 'empty':
                empty_dict[type] = cp_results[f"{type}_{evaluate}"]

    df_cov = pd.DataFrame.from_dict(cov_dict)
    df_len = pd.DataFrame.from_dict(len_dict)
    df_empt = pd.DataFrame.from_dict(empty_dict)
    df_cov.set_index(cp_results['confidence']*100, inplace=True)
    df_len.set_index(cp_results['confidence']*100, inplace=True)
    df_empt.set_index(cp_results['confidence']*100, inplace=True)

    print(df_cov)
    print(df_len)
    print(df_empt)