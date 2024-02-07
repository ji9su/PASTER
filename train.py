
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch.optim as optim
from torchvision import transforms

import sys
sys.path.append('../..')

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer


class CXR(Dataset):
  
  def __init__(self, df, transform):
    
    self.img_paths = list(df["img_path"])
    self.reports = list(df["REPORT"])
    
    self.transform = transform

  def __len__(self):
    
    return len(self.img_paths)
  
  def __getitem__(self, index):
    
    text = self.reports[index]
    
    img_path = self.img_paths[index]
    
    image = Image.open(img_path)
    
    image = self.transform(image)
    
    data = {'img':image, "txt":text, "img_path":img_path}
    
    return data

def load_data(file_path, args, batch_size=4, column='report', verbose=False): 
    
    if torch.cuda.is_available():
        device = "cuda:0"
        print('Using CUDA.')
    else:
        device = "cpu"
        print('Using cpu.')
    
    device = torch.device(device)
    
    transform = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(256),
      transforms.Grayscale(num_output_channels=3),
      transforms.ToTensor()
    ])
    
    data = pd.read_csv(file_path)

    train_data = data.loc[data["DATASET"].isin(["train"]),:]
    valid_data = data.loc[data["DATASET"].isin(["valid"]),:]

    train_dataset = CXR(train_data, transform = transform)
    valid_dataset = CXR(valid_data, transform = transform)

    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample['img'][0])
            plt.show()
            print(i, sample['img'].size(), sample['txt'])
            if i == 3:
                break
    
    loader_params = {'batch_size':batch_size, 'shuffle': False, 'num_workers': 0, "drop_last":True, "collate_fn":None, "sampler":None, "pin_memory":True}
    train_loader = DataLoader(train_dataset, **loader_params)
    
    loader_params = {'batch_size':batch_size, 'shuffle': False, 'num_workers': 0, "drop_last":False, "collate_fn":None, "sampler":None, "pin_memory":True}
    valid_loader = DataLoader(valid_dataset, **loader_params)
    
    return [train_loader, valid_loader], device
    
def load_clip(model_path=None, pretrained=False, context_length=77, device = None):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model 
    architecture. 
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * pretrained (optional) - if True, will load the pretrained 
        CLIP model
        * context_length (optional) - length of the maximum number of 
        tokens that can be inputted into the CLIP model
    '''

    params = {
        'embed_dim':512,
        'image_resolution': 256,
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 32,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    if pretrained: 
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else: 
        model = CLIP(**params)
        print("Loaded in clip model.")

    # if a model_path is provided, load in weights to backbone
    if model_path != None:
        model.load_state_dict(torch.load(model_path, map_location = device))
        # change token_lenght
        # model.context_length = 128
        # model.positional_embedding = nn.Parameter(torch.empty(model.context_length, 512))
        # nn.init.normal_(model.positional_embedding, std = 0.01)
        # model.positional_embedding.data[:state_dict['positional_embedding'].shape[0]] = state_dict["positional_embedding"]
        
    return model

def preprocess_text(texts, model):
    # if model.context_length is None:
    #     model = model.module
        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def make(config, cxr_filepath, txt_filepath, model_path=None): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer. 
    
    args: 
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
    '''
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=config.batch_size, pretrained=config.pretrained, column=config.column)
    model = load_clip(model_path=model_path, pretrained=config.pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    # todo: incorporate - torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return model, data_loader, device, criterion, optimizer

def train_main(cxr_filepath, txt_filepath, hyperparams, output_path, model_path=None, pretrained=False): 
    '''
    args: 
        * cxr_filpath- str filepath to cxr images
        * txt_filepath- str filepath to text reports
        * hyperparams- dictionary with the following hyperparams:
        `batch_size`, `criterion`, `learning_rate`, `momentum`, `epochs`
        * output_path- str filepath to where the trained model will be saved
        * model_path- str filepath to model that will be used as baseline model for training. 
        If not provided, a model will be trained from scratch
        * pretrained- whether or not the clip model was pretrained with generic images 
    This function is the main train function for CXR-CLIP. 
    '''
    
    # unpack `hyperparams`
    batch_size = hyperparams['batch_size']
    criterion = hyperparams['criterion']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    epochs = hyperparams['epochs']
    
    # load input cxr + report data
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=batch_size, pretrained=pretrained)
    model = load_clip(model_path=model_path, pretrained=pretrained)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_clip(model, data_loader, device, criterion, optimizer, epochs, output_path)
    
    return model
