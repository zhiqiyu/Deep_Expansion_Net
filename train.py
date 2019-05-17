'''train the model'''
import argparse
import logging
import os
import glob

from model.data_loader import *
from model.x2net import x2Net
from model.x3net import x3Net
from model.x4net import x4Net
from model.loss_fn import loss_fn
from model.metrics import metrics

from evaluate import evaluate
import utils

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/x3/', help="Directory containing the dataset")
parser.add_argument('--model', default='x3net', help='The model to train and test')
parser.add_argument('--model_dir', default='experiments/x3net/', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  

def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    
    model = model.to(params.device)

    for i, (X_batch, y_batch) in enumerate(dataloader):
        # move to GPU if available
        # if params.cuda:
        X_batch, y_batch = X_batch.to(params.device), y_batch.to(params.device)
        
        # compute model output and loss
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # clear previous gradients, compute gradients of all variables w.r.t. loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # move output and ground truth to cpu, convert to numpy arrays
            y_pred = y_pred.detach().cpu()
            y_batch = y_batch.detach().cpu()

            # compute all metrics on this batch
            summary_batch = {metric : metrics[metric](y_pred, y_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss.item())
    
    # compute mean of all metrics in summary
    metrics_mean = {metric : np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None, lr_scheduler=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
        lr_scheduler: (optim.lr_scheduler) learning rate scheduler 
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_metric = 0    # we use mse for metric here, so need to set the initial to a large number

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # learning rate scheduler
        if lr_scheduler:
            lr_scheduler.step()

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_metric = val_metrics['psnr']
        is_best = val_metric >= best_val_metric

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best validation metric")
            best_val_metric = val_metric

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)



if __name__ == '__main__':

    # set thread number to 1
    torch.set_num_threads(1)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    params.device = torch.device("cuda:2" if params.cuda else "cpu")
    # cudnn.benchmark = True

    # Set the random seed for reproducible experiments
    torch.manual_seed(590)
    if params.cuda: torch.cuda.manual_seed(590)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = fetch_dataloaders(['training', 'validation', 'test'], args.data_dir, params)
    train_dl = dataloaders['training']
    val_dl = dataloaders['validation']


    logging.info("- done.")

    # Define the model and optimizer
    model_name = args.model
    if model_name == 'x2net':
        model = x2Net().to(params.device)
    elif model_name == 'x3net':
        model = x3Net().to(params.device)
    elif model_name == 'x4net':
        model = x4Net().to(params.device)
    else:
        print('not implemented')
        exit()
    
    # use 2 GPUs for training
    if params.cuda:
        model = nn.DataParallel(model, device_ids=[2,3])


    # defin optimizer   
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # learning rate scheduler  
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 30, 60], gamma=0.1)

    # # fetch loss function and metrics
    # loss_fn = net.loss_fn
    # metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir, args.restore_file, lr_scheduler=scheduler)



