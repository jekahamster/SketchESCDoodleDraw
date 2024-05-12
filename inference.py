import os 
import random 
import datetime 
from opts import parse_inference_opt

import numpy as np
import torch 
import torch.nn as nn 

from torch.utils.data import DataLoader
from dataset import QuickDrawDataset, DoodleDataset
from models.sketch_transformer import ViTForSketchClassification



def main():
    opt = parse_inference_opt()
    dataset_path = opt["dataset_path"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = QuickDrawDataset(dataset_path, mode="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ViTForSketchClassification.from_pretrained(
        opt['pretrain_path'], 
        opt, 
        labels_number = dataset.num_categories(), 
        attention_probs_dropout_prob = opt['attention_dropout'], 
        hidden_dropout_prob = opt['embedding_dropout'], 
        use_mask_token = opt['mask']
    ).to(device)

    model.eval()
    with torch.no_grad():
        pass


if __name__ == "__main__":
    main()