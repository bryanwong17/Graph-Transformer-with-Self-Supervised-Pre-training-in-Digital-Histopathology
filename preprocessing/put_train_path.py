import os
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd

# D:0
# M:1
# N:2

def main():

    columns = ["path to patches"]
    SOURCE_ROOT = "E:/sample_3000"
    SUBSET = sorted(["train", "val", "test"])
    CLASSES = sorted(["LUAD", "LUSC"])

    for subset in SUBSET:
        with open(f"E:/TCGA_{subset}_sample_3000.csv", 'w', newline="") as f:
            writer = csv.writer(f)
            # write the name of column
            writer.writerow(columns)
            for _class in CLASSES:
                source_path = f"{SOURCE_ROOT}/{subset}/{_class}"
                slide_files = os.listdir(source_path)
                for s in slide_files:
                    patch_files = os.listdir(os.path.join(source_path, s))
                    for p in patch_files:
                        each_patch = os.path.join(source_path, s, p)
                        writer.writerow([each_patch])

if __name__ == "__main__":
    main()

