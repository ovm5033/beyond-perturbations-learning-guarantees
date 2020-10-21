from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms, models

from helper import *

parser = argparse.ArgumentParser(description='Selective Classification Robustness')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=76, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--arch', type=str, default='ResNet18')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--dataset', default='CIFAR10')
parser.add_argument('--task', type=str, default='distinguisher')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--weighted_train', type=float, default=1.0)

args = parser.parse_args()

# Create directory if it doesn't exist
if not os.path.isdir('./checkpoints/'):
    os.mkdir('./checkpoints/')

checkpoint_dir='./checkpoints/dset_{}_arch_{}_epochs_{}_learnrate_{}/'.format(args.dataset, args.arch, args.epochs, args.lr)

# Create directory if it doesn't exist
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
#kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

#Setup logging format
setup_logger('log', f'{checkpoint_dir}log.log', logging.INFO, '%(asctime)s : %(message)s')
log = logging.getLogger('log')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(data)

        # do weighted training for distinguisher based on class
        if args.task == 'distinguisher':
            # train dist examples will be weighted with args.weighted_train
            # test dist examples will be weighted with 1.0
            # weights
            weights = torch.tensor([args.weighted_train, 1.0]).to(device)
            loss = torch.nn.CrossEntropyLoss(weight=weights)(outputs, target)
        
        if args.task == 'classifier':
            loss = torch.nn.CrossEntropyLoss()(outputs, target)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    log.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    """ schedule: 0.1 for [1, 56], 0.01 for [57, 71] and 0.001 for [72, 85]  """
    lr = args.lr
    if epoch == 57 or epoch == 72:
        lr = args.lr * 0.1

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_classifier_distinguisher():

    # load dataset, load_data_distinguishing
    if args.task == 'distinguisher':
        if args.dataset == 'EMNIST-Adv':
            classifier = load_model(checkpoint_dir, 'classifier', args.arch, device, args.num_classes)
            train_loader, test_loader = load_data_distinguishing(args.dataset, args.batch_size, classifier)
        else:
            train_loader, test_loader = load_data_distinguishing(args.dataset, args.batch_size)

    elif args.task == 'classifier':
        train_loader, test_loader = load_data(args.dataset, args.batch_size, task='train_dist')

    checkpoint = None
    # Check if resume is needed
    if args.resume:
        # Load checkpoint.
        log.info('==> Resuming from checkpoint..')
        #assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'

        resume_file = f'{checkpoint_dir}_{args.task}.pth'
        if os.path.isfile(resume_file):
            checkpoint = torch.load(resume_file)
        else:
            log.info('No checkpoint found. Starting from scratch.')

    # Initialize model architecture and optimizer
    if args.task == 'distinguisher':
        # Load model
        model = model_architecture(args.arch, device, num_classes=2)
        model = model.to(device)

    if args.task == 'classifier':
        # we start training a classifier from scratch
        model = model_architecture(args.arch, device, num_classes=args.num_classes)
        model = model.to(device)

    # model = model_architecture(args.arch, device)
    # we might want to consider changing hyperparams
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay)

    # Resume training from where we left off
    start_epoch = 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['opt'])
        start_epoch = checkpoint['epoch']+1

    for epoch in range(start_epoch, args.epochs + 1):
        log.info('Starting Epoch %d' % (epoch))

        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # train model for one epoch
        train(args, model, device, train_loader, optimizer, epoch)

        # Save a checkpoint (for resuming later)
        state = {
            'model': model.state_dict(),
            'epoch': epoch,
            'opt':optimizer.state_dict(),
        }
        save_checkpoint(state, args.task, checkpoint_dir)

    # store the final model
    log.info('Saving Model...')
    torch.save(model.state_dict(), f'{checkpoint_dir}_{args.task}_weight_{args.weighted_train}_final.pth')

    # Evaluate model after training
    model = model.to(device)
    test_loss, test_accuracy = eval_test(model, device, test_loader)
    log.info('Evaluation Results - %s' % (args.task))
    log.info('Test Accuracy: %f' % (test_accuracy))
    log.info('Test Loss: %f' % (test_loss))

    if args.task == 'classifier':
        # also evaluate on Q test
        # Evaluate model after training

        _, Q_test_loader = load_data(args.dataset, args.batch_size, task='test_dist', classifier=model)

        model = model.to(device)
        test_loss, test_accuracy = eval_test(model, device, Q_test_loader)
        log.info('Q-Test Evaluation Results - %s' % (args.task))
        log.info('Q Test Accuracy: %f' % (test_accuracy))
        log.info('Q Test Loss: %f' % (test_loss))

# Main
if args.task == 'distinguisher' or args.task == 'classifier':
    train_classifier_distinguisher()
