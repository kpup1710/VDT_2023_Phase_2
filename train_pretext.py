import torch
from tqdm import tqdm
import numpy as np
from optimizer import *
import os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def loss_batch(model, loss_func, x1, x2, opt=None, metric=None):
    # Generate predictions
    y1,_ = model(x1)
    y2,_ = model(x2)
    # Calculate loss
    # print(max(yb))
    loss = loss_func(y1, y2)

    if opt is not None:
        # Compute gradients
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients
        opt.zero_grad()
    metric_result = None
    if metric is not None:
        # Compute the metric
        metric_result = metric(y1, y2)
    return loss.item(), len(x1), metric_result


def evaluate(model, loss_func, valid_dl, metric=None):
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_func, x1, x2, metric=metric)
                   for x1, x2 in valid_dl]
        # Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        # Total size of the data set
        total = np.sum(nums)
        # Avg, loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = 0
        if metric is not None:
            # Avg of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train_pretext(epochs, model, loss_func, train_dl, valid_dl, opt_fn=None, lr=None, metric=None, expt_name='cedar', PATH=''):
    train_losses, val_losses, val_metrics = [], [], []
    torch.cuda.empty_cache()    
    # Instantiate the optimizer
    if opt_fn is None:
        opt_fn = torch.optim.SGD

    params, param_names = [], []
    for name, param in model.named_parameters():
        params.append(param)
        param_names.append(name)

    parameters = [{'params' : params, 'param_names' : param_names}]
    opt = opt_fn(parameters, lr = lr)
    sched = LinearWarmupCosineAnnealingLR(opt, 10, epochs)
    max_val_loss = 1e4
    for epoch in range(epochs):
        # Training
        model.train()
        for x1, x2 in tqdm(train_dl):
            train_loss,_,_ = loss_batch(model, loss_func, x1, x2, opt)

        # Evaluation
        model.eval()
        result = evaluate(model, loss_func=loss_func, valid_dl=valid_dl, metric=metric)
        val_loss, total, val_metric = result
        if max_val_loss < val_loss:
            print("saving model")
            torch.save({'model_state_dict' : model.state_dict(),
                'optim_state_dict' : opt.state_dict(),
                'scheduler_state_dict' : sched.state_dict(),
                'epochs' : epoch+1},
                PATH+ f'/model_{expt_name}.pt')
            print("saved model")
        sched.step(val_loss)
        # Record the loss and metric
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        # Print progress
        if metric is None:
            messages = 'Epoch [{} / {}], train_loss: {:4f}, val_loss:{:4f}'\
                .format(epoch + 1, epochs, train_loss, val_loss)
        else:
            messages = 'Epoch [{} / {}], train_loss: {:4f}, val_loss:{:4f}, val_{}: {:4f}'\
                  .format(epoch + 1, epochs, train_loss, val_loss, metric.__name__, val_metric)
        logger.info(messages)
    return train_losses, val_losses, val_metrics

