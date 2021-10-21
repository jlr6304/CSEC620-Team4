import pandas as pd
import numpy as np 
import os

def get_malware_samples():
    
    malware_df = pd.read_csv('data/sha256_family.csv', delimiter=',')
    
    files = set(os.listdir("data\\feature_vectors\\feature_vectors"))

    malware_files = set(malware_df['sha256'])
    all_benign_files = list(files - malware_files) 
    
    benign_files = np.random.choice(all_benign_files, 100, replace=False)

    malware_df = malware_df.loc[:100, ]
    benign_df = pd.DataFrame({
        'sha256': benign_files,
        'family': np.full(len(benign_files), fill_value='Benign')
    })

    files_df = pd.concat((benign_df, malware_df), axis=0)
    
    print(len(files_df))
    
    lines = {}
    unique_features = set()

    for k, file in enumerate(files_df['sha256']):
        
        with open("data\\feature_vectors\\feature_vectors\\" + file) as f:
            lines[file] = f.readlines()
            
            unique_features = unique_features.union(set(lines[file]))


if __name__ == '__main__':
    get_malware_samples()