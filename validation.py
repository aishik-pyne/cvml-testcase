from typing import Dict
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchmetrics import Accuracy, ConfusionMatrix
from tqdm import tqdm

from models.bilstm_pool_bilstm import BiLSTM_Pool_LSTM

def validate(model: BiLSTM_Pool_LSTM,
            criterion,
            loader: DataLoader,
            num_classes: int,
            actions_dict: Dict,
            writer: SummaryWriter = None,
            device: str = "cpu",
            global_step: int = None):

    model.eval()

    # Define Evaluation Metrics
    perclass_acc_metric = Accuracy(average='none', num_classes=num_classes).to(device)
    global_acc_metric = Accuracy(num_classes=num_classes).to(device)
    confmat_metric = ConfusionMatrix(num_classes=num_classes).to(device)
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Validate Epoch {global_step} \t")
        # Iterate over all vids in the validation dataset
        for X, y, X_lengths in pbar:
            X = X.float().to(device)
            y = y.long().to(device)
            
            # Forward Loss
            pred = model(X, X_lengths)

            # Loss Objection
            loss = criterion(pred.view(-1, num_classes), y.view(-1))
            total_loss += loss.detach()

            mask = torch.concat(
              [torch.BoolTensor([True] * X_lengths[i] + [False] * (len(X[i]) - X_lengths[i])) for i in range(len(X))],
              dim=0).to(device)

            perclass_acc_metric.update(
              torch.masked_select(torch.argmax(pred.view(-1, num_classes), dim=-1), mask),
              torch.masked_select(y.view(-1), mask))
            global_acc_metric.update(
              torch.masked_select(torch.argmax(pred.view(-1, num_classes), dim=-1), mask),
              torch.masked_select(y.view(-1), mask))
            confmat_metric.update(
              torch.masked_select(torch.argmax(pred.view(-1, num_classes), dim=-1), mask),
              torch.masked_select(y.view(-1), mask))

    total_loss = total_loss / len(loader)
    total_acc = global_acc_metric.compute().detach().cpu()
    perclass_acc = perclass_acc_metric.compute().detach().cpu()
    total_conf = confmat_metric.compute().detach().cpu()
    
    
    if writer and global_step:
        # Log class wise accuracy and loss
        writer.add_scalar("Loss/test", total_loss, global_step)
        writer.add_scalar("Accuracy/test", total_acc, global_step)

        # Log class wise accuracy
        plt.figure(figsize = (10,7))
        fig_ = sns.barplot(x=list(actions_dict.keys()), y=perclass_acc.numpy()).get_figure()
        plt.xticks(rotation=70)
        plt.close(fig_)
        writer.add_figure("Class wise accuracy", fig_, global_step)

        # Log Confusion matrix
        df_cm = pd.DataFrame(total_conf.numpy(), index = list(actions_dict.keys()), columns=list(actions_dict.keys()))
        plt.figure(figsize = (15,10))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        writer.add_figure("Confusion matrix with annot", fig_, global_step)

        # Log Confusion matrix
        df_cm = pd.DataFrame(total_conf.numpy(), index = list(actions_dict.keys()), columns=list(actions_dict.keys()))
        plt.figure(figsize = (15,10))
        fig_ = sns.heatmap(df_cm, annot=False, cmap='Spectral').get_figure()
        plt.close(fig_)
        writer.add_figure("Confusion matrix", fig_, global_step)
        
        # Plot sample prediction
        fig_, axs = plt.subplots(nrows=len(X), ncols=2, figsize=(15, 8*len(X)))
        for i, y_ in enumerate(y):
          axs[i][0].set_ylim([0, num_classes+1])
          sns.scatterplot(x=np.arange(X_lengths[i]), y=y_[:X_lengths[i]].cpu(), ax=axs[i][0])
        for i, X in enumerate(X):
          axs[i][1].set_ylim([0, num_classes+1])
          sns.scatterplot(x=np.arange(X_lengths[i]), y=torch.argmax(pred[i][:X_lengths[i]], dim=-1).cpu(), ax=axs[i][1])
        plt.close(fig_)
        writer.add_figure("Prediction Sequence", fig_, global_step)
        

    model.train()

    return total_acc, total_loss, total_conf


