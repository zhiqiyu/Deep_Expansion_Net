"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils

from model.x2net import x2Net
from model.x3net import x3Net
from model.x4net import x4Net
from model.loss_fn import loss_fn
from model.metrics import metrics
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/x2/', help="Directory containing the dataset")
parser.add_argument('--model', default='x2net', help="The model to use.")
parser.add_argument('--model_dir', default='experiments/x2net', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for X_batch, y_batch in dataloader:

        # move to GPU if available
        # if params.cuda:
        X_batch, y_batch = X_batch.to(params.device), y_batch.to(params.device)
        
        # compute model output
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        y_pred = y_pred.detach().cpu()
        y_batch = y_batch.detach().cpu()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](y_pred, y_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.5f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available
    params.device = torch.device("cuda:2" if params.cuda else "cpu")

    # Set the random seed for reproducible experiments
    torch.manual_seed(590)
    if params.cuda: torch.cuda.manual_seed(590)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloaders(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
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
    
    if params.cuda: 
        model = torch.nn.DataParallel(model, device_ids=[2,3])
    
    # loss_fn = net.loss_fn
    # metrics = net.metrics
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
