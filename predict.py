
import os
import re
import pandas as pd
import numpy as np
import json
from PIL import Image
from typing import List, Dict, Optional

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from zero_shot import load_clip, run_single_prediction
from eval import evaluate

class CXR(Dataset):
  
  def __init__(self, df, transform):
    
    self.img_paths = list(df["img_path"])
    self.transform = transform

  def __len__(self):
    
    return len(self.img_paths)
  
  def __getitem__(self, index):
    
    img_path = self.img_paths[index]
    
    image = Image.open(img_path)
    
    image = self.transform(image)
    
    data = {'img':image, "img_path":img_path}
    
    return data
  

# ------- SETTING ------ #
is_feature_extraction = True
is_zero_shot = True


# ------- LABELS ------ #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = ['left ventricular dysfunction']

device = torch.device("cuda:0")

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
pair_template: Dict[str, str] = {"POS":"{classname}", "NEG":"no {classname}"}


# ---- LOAD DATA ----- #
data = pd.read_csv('./data/example_data.csv')
test_data = data.loc[data["DATASET"].isin(["test"]),:]


transform = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(256),
      transforms.Grayscale(num_output_channels=3),
      transforms.ToTensor()
    ])


train_dataset = CXR(test_data, transform = transform)

dataloader = DataLoader(train_dataset, batch_size = 2, shuffle=False)

model_paths = ["./result/checkpoints/best_model.pth"]


for model_path in model_paths:
  
  model = load_clip(
    model_path = model_path, 
    pretrained = True, 
    context_length = 256,
    device = device
    )
  
  model.eval()
  # ---- FEATURE EXTRACTION ----- #
  if is_feature_extraction:
    img_feat, img_path_list = run_single_prediction(cxr_labels, model, dataloader, device,
                                                    softmax_eval = True, context_length = 256)

    image_feat_df = pd.DataFrame({"img_path": img_path_list, **{"img_feat:" + str(index): img_feat[:,index] for index in range(512)}})

    image_feat_df.to_csv("./result/img_feat.csv", index = False)
  
  # ---- ZERO SHOT ----- #
  if is_zero_shot:

    zero_shot_df = pd.DataFrame()
    
    for key, prompt in pair_template.items():
      
      prompt_pred, img_path_list = run_single_prediction(cxr_labels, model, dataloader, device,
                                                         template = prompt, softmax_eval = True, 
                                                         context_length = 256,
                                                         is_feature_extraction = False)
                                                         
      result_df = pd.DataFrame({"img_path": img_path_list,
                                **{cxr_labels[index] + ":" + key:prompt_pred[:,index] for index in range(len(cxr_labels))}})
      
      if zero_shot_df.empty:
        zero_shot_df = result_df
      else:
        zero_shot_df = pd.merge(zero_shot_df, result_df, on = "img_path")
        
    # ---- evaluate ----- #
    for cxr_label in cxr_labels:
      zero_shot_array = zero_shot_df.loc[:,[cxr_label + ":POS", cxr_label + ":NEG"]].to_numpy()
      
      pos_pred = zero_shot_array[:,0]
      neg_pred = zero_shot_array[:,1]
      
      sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
      y_pred = np.exp(pos_pred) / sum_pred
      
      y_true = np.array(test_data[cxr_label])
      roc_auc = evaluate(y_pred, y_true, cxr_label)
      print(cxr_label, " AUC: ", roc_auc)



