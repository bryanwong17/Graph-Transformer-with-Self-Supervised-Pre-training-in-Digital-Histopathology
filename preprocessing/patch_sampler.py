import pandas as pd
import random
import os
import shutil
import csv

path = f'C:/Users/user/Documents/Team_Project_ML/new_patch_512'
d = {}
for a, sub_folder in enumerate(os.listdir(path)):
    d[sub_folder] = {}
    for index, clas in enumerate(os.listdir(os.path.join(path,sub_folder))):
        d[sub_folder][clas] = {}
        for idx, slide_folder in enumerate(os.listdir(os.path.join(path,sub_folder,clas))):
            d[sub_folder][clas][slide_folder] = []
            for a, patch in enumerate(os.listdir(os.path.join(path,sub_folder,clas,slide_folder))):
                p = os.path.join(path,sub_folder,clas,slide_folder,patch)
                d[sub_folder][clas][slide_folder].append(p)

csv_all = []
n_sample = 3000
for sub_folder in d:
    csv_sub = []
    for clas in d[sub_folder]:
        for slide in d[sub_folder][clas]:
            if len(d[sub_folder][clas][slide]) < n_sample:
                n = len(d[sub_folder][clas][slide])
            else:
                n = n_sample
            sam = random.sample(d[sub_folder][clas][slide],n)
            csv_sub += sam
            csv_all += sam
        print(f'{sub_folder} - {clas} done!')
    df = pd.DataFrame(csv_sub)
    df = df.sample(frac=1)
    df.to_csv(f'C:/Users/user/Documents/Team_Project_ML/TCGA_512_{sub_folder}_sample_new_{n_sample}.csv',index=False)
df2 = pd.DataFrame(csv_all)
df2.to_csv(f'C:/Users/user/Documents/Team_Project_ML/TCGA_512_all_sample_new_{n_sample}.csv',index=False)