import torch
from torch import optim, nn
from torchvision import models, transforms
from PIL import Image

import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
import shutil
from matplotlib import pyplot as plt
import random
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import cv2

model = models.vgg16(pretrained=True)

class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 

# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])

def extract_features(path):
    img = cv2.imread(path)
    # Transform the image
    img = transform(img)
    # Reshape the image. PyTorch model reads 4-dimensional tensor
    # [batch_size, channels, width, height]
    img = img.reshape(1, 3, 448, 448)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the feature from the image
        feature = new_model(img)
    return feature.cpu().detach().numpy().reshape(-1)

root = 'D:/ML Dataset/TCGA_512_train_sample_3000.csv'
root = pd.read_csv(root)
total_patch = len(root)
data = {}
p = 0

# Iterate each image
for r in root['0']: 
    # slide = '/'.join(os.path.normpath(r).split(os.sep)[0:7])
    if r not in list(data.keys()):
        feature = extract_features(r)
        data[r] = feature
    else:
        next
    p += 1
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f"[{p}/{total_patch}] {r}: {current_time}")

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
features = np.array(list(data.values()))
features.shape

k = 32
kmeans = KMeans(n_clusters=k, random_state=22)
kmeans.fit(features)

# holds the cluster id and the images { id: [images] }
groups = {}
i = 0
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)
    i += 1
    print(f"[{i}/{len(filenames)}] {cluster}: {file}")

# function that lets you view a cluster (based on identifier)        
def view_cluster(groups,cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = Image.open(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

view_cluster(groups,10)
view_cluster(groups,27)
view_cluster(groups,28)

lst = list(zip(filenames,kmeans.labels_))
df = pd.DataFrame(lst, columns = ['path','cluster'])
# df.to_csv(f"C:/Users/user/Documents/Team_Project_ML/TCGA_512_train_cluster_k{k}.csv", index=False)

total_batch = round(len(df)/k)
# n_cluster = len(df['cluster'].value_counts())
cluster_dict = {}

for i in range(k):
    cluster_dict[i] = {} 
    cluster_dict[i]['len'] = len(df[df.cluster == i])
    cluster_dict[i]['path'] = list(df.loc[df.cluster==i,'path'])

patch_lst = []
cluster_lst = []
batch_lst = []

for j in cluster_dict.keys():
    accum = cluster_dict[j]['len']
    while accum > 0:
        for l in range(total_batch):
            m = len([b for b in batch_lst if b==l])
            if accum > 0 and m < k:
                sam = random.sample(cluster_dict[j]['path'],1)
                patch_lst.append(sam[0])
                cluster_lst.append(j)
                batch_lst.append(l)
                cluster_dict[j]['path'] = [p for p in cluster_dict[j]['path'] if p not in sam]
                accum -= 1
                m += 1
            else:
                next
            # print(f'cluster:{j},batch:{l},m:{m}')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'[{j}/{k-1}] {current_time}')

df2_lst = list(zip(patch_lst,cluster_lst,batch_lst))
df2 = pd.DataFrame(df2_lst, columns = ['path','cluster','batch'])
df2 = df2.sort_values(by=['batch'])
df2.to_csv(f"C:/Users/user/Documents/Team_Project_ML/TCGA_512_train_cluster_k{k}_batch.csv", index=False)