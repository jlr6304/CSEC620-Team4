import pandas as pd
import numpy as np 
import os

def get_samples():
    
    malware_df = pd.read_csv('data/sha256_family.csv', delimiter=',')
    
    files = set(os.listdir("data\\feature_vectors\\feature_vectors"))

    malware_files = set(malware_df['sha256'])
    all_benign_files = list(files - malware_files) 
    
    benign_files = np.random.choice(all_benign_files, 2000, replace=False)

    malware_df = malware_df.loc[:500, ]
    
    benign_df = pd.DataFrame({
        'sha256': benign_files,
        'family': np.full(len(benign_files), fill_value='Benign')
    })

    files_df = pd.concat((benign_df, malware_df), axis=0, ignore_index=True).set_index("sha256")
    
    
    lines = {}
    unique_features = set()

    print("Feature extraction")
    for k, file in enumerate(files_df.index):
        if k%1000 == 999:
            print(f'{k+1}/{len(files_df)} files treated')
        
        with open("data\\feature_vectors\\feature_vectors\\" + file) as f:
            lines[file] = [l.strip() for l in f.readlines()]
            
            unique_features = unique_features.union(set(lines[file]))
   
    print(f'number of features: {len(unique_features)}')
    
    # features_df = {}
    # for k,feature in enumerate(unique_features):
    #     if k%100 == 99:
    #         print(f'{k+1}/{len(unique_features)}')
    #     feature_vector=[]
    #     for file in files_df['sha256']:
    #         if feature in lines[file]:
    #             feature_vector.append(1)
    #         else:
    #             feature_vector.append(0)
        
    #     features_df[feature]=feature_vector

    # features_df = pd.DataFrame(features_df)
    
    # samples_df = pd.concat((files_df, features_df), axis=1)
    
    print("Matrix filling")
    features_df = pd.DataFrame(0, columns=unique_features, index = files_df.index)
    for k,file in enumerate(files_df.index):
        if k%1000 == 999:
            print(f'{k+1}/{len(files_df)} rows filled')
        
        features_df.loc[file, lines[file]] = 1
    
    # --- Method 1
    # print("Adding family")
    # features_df['family'] = files_df['family']

    # samples_df = features_df.rename(
    #     columns = dict(zip(
    #         unique_features,
    #         ["f" + str(i) for i in range(len(unique_features))]
    #     ))
    # ).reset_index(drop=True)

    # print(samples_df.sum(axis=1).sort_values())

    # print("Saving files")
    # samples_df.to_pickle('data/preprocessed_samples.pkl')

    # --- Method 2
    samples = features_df.values
    np.save('data/samples', samples)

    family = np.array(files_df['family'])
    np.save('data/family', family)


if __name__ == '__main__':
    get_samples()
