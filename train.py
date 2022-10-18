from typing import Dict
from unicodedata import name
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy
from models.bilstm_pool_bilstm import BiLSTM_Pool_LSTM

from tqdm import tqdm
from validation import validate
from utils.checkpoints import save_model_checkpoint


def train(model: BiLSTM_Pool_LSTM,
          model_name: str,
          train_loader,
          test_loader,
          criterion,
          optimizer,
          scheduler,
          num_classes: int,
          actions_dict: Dict,
          num_epochs: int = 50,
          device: str = "cpu",
          global_step=0,
          writer=None,
          force=bool):

    model.train()

    if writer is None:
        writer = SummaryWriter()

    writer.add_hparams({'num_epochs': num_epochs}, {})

    loss = torch.zeros(1)

    for e in range(num_epochs):
        pbar = tqdm(
            train_loader, desc=f"Train    Epoch {global_step} \t")

        epoch_loss = 0
        train_acc_metric = Accuracy(num_classes=num_classes).to(device)
        for X, y, lengths in pbar:
            X = X.float().to(device)
            y = y.long().to(device)
            
            # Forward Loss
            pred = model(X, lengths).view(-1)

            # Loss Objection
            loss = criterion(pred.view(-1, num_classes), y.view(-1))
            epoch_loss += loss.detach()

            mask = torch.concat(
              [torch.BoolTensor([True] * lengths[i] + [False] * (len(X[i]) - lengths[i])) for i in range(len(X))],
              dim=0).to(device)

            train_acc_metric.update(
              torch.masked_select(torch.argmax(pred.view(-1, num_classes), dim=-1), mask),
              torch.masked_select(y.view(-1), mask))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            



        # Log Metrics
        train_acc, train_loss = train_acc_metric.compute().detach(), epoch_loss/len(train_loader)
        writer.add_scalar("Loss/train", train_loss, global_step)
        writer.add_scalar("Accuracy/train", train_acc, global_step)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], global_step)

        pbar.update(
            f"Epoch {e} | Loss {train_loss} | LR = {optimizer.param_groups[0]['lr']}")
        model.log_weights_and_grads(writer=writer, global_step=global_step)

        # Validation Loop
        test_acc, test_loss, _ = validate(model,
                              criterion=criterion,
                              loader=test_loader,
                              num_classes=num_classes,
                              actions_dict=actions_dict,
                              writer=writer,
                              device=device,
                              global_step=global_step)
        print(f"Train Loss {train_loss:.3f} | Train Acc {train_acc:.3f} | Test Loss {test_loss:.3f} | Test Acc {test_acc:.3f}\n")
        scheduler.step(test_loss)
        global_step += 1

        # Checkpoint every 5 epoch
        if global_step % 10 == 0:
            save_model_checkpoint(
                name=model_name,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                force=force
            )
    return global_step
