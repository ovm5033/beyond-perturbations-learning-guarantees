from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

from helper import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd
import numpy as np

import itertools
import numpy as np
import scipy

parser = argparse.ArgumentParser(description='Selective Classification Robustness Evaluation')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--dataset', default='CIFAR10')
parser.add_argument('--num_classes', type=int, default=10)

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
print(torch.cuda.is_available())
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
#kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# checkpoint directory where model is stored
checkpoint_dir='./checkpoints/dset_{}_arch_{}_epochs_{}_learnrate_{}/'.format(args.dataset, args.arch, args.epochs, args.lr)

#Setup logging format
setup_logger('log', f'{checkpoint_dir}log.log', logging.INFO, '%(asctime)s : %(message)s')
log = logging.getLogger('log')


def eval_test(model, device, test_loader, permute=False):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if permute:

                # randomly shuffle the pixels in each image in the batch
                idx_perm = torch.randperm(3*32*32)
                data = data.view(len(data), -1)[:, idx_perm].view(len(data), 3,32,32)
            
            #random_noise = torch.FloatTensor(*data.shape).uniform_(-0.1, 0.1).to(device)
            #data = data + random_noise
            
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy

def compute_probs_scores(classifier, distinguisher, data_loader):

    if classifier is not None:
        classifier.eval()

    if distinguisher is not None:
        distinguisher.eval()

    scores = []
    wrongs = []

    for data,target in data_loader:
        data, target = data.to(device), target.to(device)

        if classifier is not None:
            output = classifier(data)
            pred = output.max(1, keepdim=True)[1]
            wrong = ~pred.eq(target.view_as(pred))
            wrongs.extend(wrong.squeeze().tolist())

        if distinguisher is not None:
            # compute distinguisher probs for the data
            score = F.softmax(distinguisher(data), dim=1)
            
            # check the score for the test distribution (i.e. Q)
            scores.extend( score[:,1].squeeze().tolist() )

    return (np.array(scores), np.array(wrongs))


def compute_correlation(classifier, distinguisher, data_loader):

    scores, wrongs = compute_probs_scores(classifier, distinguisher, data_loader)

    return np.corrcoef(wrongs, scores)[0, 1]

def plot_tradeoff_exp(classifier, distinguisher, train_data_loader, test_data_loader, dataset, plt, plot_max_p_rej=0.5):


    # Compute classifier mistakes and distinguisher scores on the test splits of P and Q.
    P_scores , P_wrons = compute_probs_scores(classifier, distinguisher, train_data_loader)
    Q_scores, Q_wrongs = compute_probs_scores(classifier, distinguisher, test_data_loader)

    thresholds = np.linspace(0.0, 1, 100)
    # compute average distinguisher score for P that is greater than threshold tau
    p_rejs = [np.mean(P_scores>tau) for tau in thresholds]
    # restrict to the subset with average distinguisher score for P < plot_max_p_rej
    thresholds = [t for t, r in zip(thresholds, p_rejs) if r<plot_max_p_rej]
    # compute the average score on the restricted thresholds
    p_rejs = [np.mean(P_scores>tau) for tau in thresholds]

    # compute the average distinguisher score for Q that is greater than threshold tau
    rejs = [np.mean(Q_scores>tau) for tau in thresholds]
    # compute average error of classifier on Q, based on distinguisher scores <= threshold tau
    normalized_errs = [np.mean(Q_wrongs[Q_scores<=tau]) for tau in thresholds]

    plt.figure()
    plt.plot(*zip(*[(r, e) for r, e in zip(rejs, normalized_errs) if not np.isnan(e)]), label="Error on selected Q (normalized)")
    plt.plot(rejs, p_rejs, label="Fraction of $P$ rejected")
    plt.xlabel("Fraction of $Q$ rejected")
    plt.legend()

    plt.savefig(f'{dataset}_tradeoff_exp.png')
    return plt

def main():

    # load classifier
    classifier = load_model(checkpoint_dir, 'classifier', args.arch, device, args.num_classes)
    log.info('Finished Loading Classifier')

    # load distinguisher
    distinguisher = load_model(checkpoint_dir, 'distinguisher', args.arch, device, num_classes=2)
    log.info('Finished Loading Distinguisher')

    # Load P, Q train/test splits
    tr_loader_train_dist, te_loader_train_dist = load_data(args.dataset, args.test_batch_size, classifier=classifier, task='train_dist')
    tr_loader_test_dist, te_loader_test_dist = load_data(args.dataset, args.test_batch_size, classifier= classifier, task='test_dist')

    # Plot
    plot_tradeoff_exp(classifier, distinguisher, te_loader_train_dist, te_loader_test_dist, args.dataset, plot_max_p_rej=0.5)


if __name__ == '__main__':
    main()
