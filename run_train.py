import os
import argparse
from tqdm import tqdm

import json
import numpy as np

import torch
from torch import nn
import torch.optim as optim

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./data/example_data.csv', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default="./result/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="checkpoints")
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0): 
    # make the model, data, and optimization problem

    model, data_loader, device, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config)

    if verbose: 
        print(model)
    return model

def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(config.file_path, args = config, batch_size=config.batch_size, column="impression")
    model = load_clip(pretrained=pretrained, context_length=config.context_length, device = device)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    return model, data_loader, device, criterion, optimizer

def train(model, loader, device, criterion, optimizer, config): 
    
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
    train_loader = loader[0]
    valid_loader = loader[1]
    # Run training
    total_batches = len(train_loader) * config.epochs
    Total_train_loss = []
    Total_valid_loss = []

    for epoch in range(config.epochs):
        
        running_loss = 0.0 # running loss over batch
        train_loss_list = []
        valid_loss_list = []
        model.train()
        for data in tqdm(train_loader):
            # get the images
            images = data['img']
            
            texts = data['txt']
            texts = preprocess_text(texts, model)
            
            # perform step for a single batch
            train_loss = train_batch(images, texts, model, device, criterion, optimizer)
            train_loss_list.append(train_loss.item())
        
        model.eval()
        for data in tqdm(valid_loader):
            
            images = data['img']
            
            texts = data['txt']
            texts = preprocess_text(texts, model)
            
            # perform step for a single batch
            valid_loss = train_batch(images, texts, model, device, criterion, optimizer, is_valid = True)
            running_loss += valid_loss.item()
            valid_loss_list.append(valid_loss.item())
        
        Total_train_loss.append(np.mean(np.array(train_loss_list)))
        Total_valid_loss.append(np.mean(np.array(valid_loss_list)))
        
        logger_dict = {"train_loss":Total_train_loss, "valid_loss":Total_valid_loss}
        
        json_object = json.dumps(logger_dict)
        with open(config.save_dir + "logger.json", "w") as outfile:
            outfile.write(json_object)
        
        print("Epoch:{i} Total train_loss:{loss:.3f}".format(i = epoch, loss = np.mean(np.array(Total_train_loss))))
        print("Epoch:{i} Total valid_loss:{loss:.3f}".format(i = epoch, loss = np.mean(np.array(Total_valid_loss))))    
        print("Saved model")
        model_path = model_save_dir + "/checkpoints_{epoch}.pth".format(epoch = epoch)
        save(model, model_path)

                
def train_batch(images, texts, model, device, criterion, optimizer, is_valid = False):
    
    images, texts = images.to(device), texts.to(device)
    
    # Forward pass ➡
    logits_per_image, logits_per_text = model(images, texts)

    # Create labels
    batch_size = images.shape[0]
    clip_labels = torch.arange(batch_size).to(device)

    # Compute loss
    loss_img = criterion(logits_per_image, clip_labels)
    loss_txt = criterion(logits_per_text, clip_labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss
    
    if is_valid:
        return loss
    else:
        # Backward pass ⬅
        optimizer.zero_grad()
        loss.backward()
        
        # Step with optimizer
        optimizer.step()
            
        return loss
    
def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
    

