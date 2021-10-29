import pandas as pd
import numpy as np 
import os

def get_samples(n_benign=20000):
    """
    ### Function that preprocesses the raw data into a dataset of the features of malwares & benign samples 
        n_benign corresponds to the number of benign sample to extract
    
    Note: this function creates 3 files
        - `data/samples.npy` contains the features information for each samples
        - `data/family.npy` contains the family information for each samples (in the same order as `samples.npy`)
        - `data/features.npy` contains the name of the features
    """
    # --- Import malwares information
    malware_df = pd.read_csv('data/sha256_family.csv', delimiter=',')
    
    files = set(os.listdir("data\\feature_vectors")) # list all the files

    # --- Separate benign and malware files
    malware_files = set(malware_df['sha256'])
    all_benign_files = list(files - malware_files) 
    benign_files = np.random.choice(all_benign_files, n_benign, replace=False) # random subset of the benign files

    # --- Merge the selected files (malware & benign) into a single dataframe
    
    benign_df = pd.DataFrame({
        'sha256': benign_files,
        'family': np.full(len(benign_files), fill_value='Benign')
    })

    files_df = pd.concat((benign_df, malware_df), axis=0, ignore_index=True).set_index("sha256")
    
    
    # --- Identify the unique features
    print("Feature extraction")

    lines = {} # lines for each file
    unique_features = set() # set of all the features found

    for k, file in enumerate(files_df.index):
        if k%1000 == 999:
            print(f'{k+1}/{len(files_df)} files treated')
        
        with open("data\\feature_vectors\\feature_vectors\\" + file) as f:
            lines[file] = [l.strip() for l in f.readlines()]
            
            unique_features = unique_features.union(set(lines[file]))
   
    print(f'number of features: {len(unique_features)}')

    # --- Fill the dataset for each file by identifying the features that are present in each file
    print("Feature filling")
    features_df = pd.DataFrame(0, columns=unique_features, index = files_df.index)
    for k,file in enumerate(files_df.index):
        if k%1000 == 999:
            print(f'{k+1}/{len(files_df)} rows filled')
        
        features_df.loc[file, lines[file]] = 1

    # --- Exportation of the preprocessed data
    # Feature values of the samples
    samples = features_df.values
    np.save('data/samples', samples)

    # Family of the samples
    family = np.array(files_df['family'])
    np.save('data/family', family)

    # Feature names 
    features = features_df.columns
    np.save('data/features', features)



# Run to proceed to a preprocessing of the data
if __name__ == '__main__':
    get_samples(n_benign=10000)
