import argparse
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shutil
import logging
import emnist
from arch import *
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, train_test_split
import random

def reseed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def rand(maxi=10**9):
    return np.random.randint(maxi)

def extract(train, test, chars,  emnist_inv_map, emnist_map, num_per_class=None):
    ids = {emnist_inv_map[c] for c in chars}
    ans = [(x, chars.index(emnist_map[y])) for x, y in zip(train, test) if y in ids]
    if num_per_class is not None:
        random.shuffle(ans)
        counts = {}
        def helper(z):
            x, y = z
            counts[y] = counts.setdefault(y, 0) + 1
            return counts[y]<=num_per_class
        ans = list(filter(helper, ans))
    return np.array([x for x, y in ans]), np.array([y for x, y in ans])            

def load_data_distinguishing(dataset, batchsize, classifier=None):

    # This function prepares the datasets for the distinguishing task
    # Examples from P are labeled with 0
    # Examples from Q are labeled with 1

    if dataset == 'EMNIST-Adv':

        reseed(0)

        emnist_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        emnist_inv_map = {c: i for i, c in enumerate(emnist_map)}
        emnist_X_train, emnist_y_train = emnist.extract_training_samples('byclass')
        emnist_X_test, emnist_y_test = emnist.extract_test_samples('byclass')
        emnist_X = np.vstack([emnist_X_train, emnist_X_test])
        emnist_y = np.hstack([emnist_y_train, emnist_y_test])
        emnist_X.shape, emnist_y.shape

        lower_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].islower()])
        upper_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].isupper()])

        N_per_class = 10*1000, 3*1000
        chars = "".join([c for c, l in lower_counts.most_common() if l>=N_per_class[0] and upper_counts[c.upper()]>N_per_class[1]])
        CHARS = chars.upper()
        min_counts = min(lower_counts[c] for c in chars), min(upper_counts[c] for c in CHARS)

        print(chars, CHARS, len(chars), min_counts)

        # P is lowercase letters
        PX_all, Py_all = extract(emnist_X, emnist_y, chars, emnist_inv_map, emnist_map, num_per_class=N_per_class[0])

        # Save some P examples for later use in Q
        n_donate = 3000*len(chars) # donate from P to Q
        assert n_donate < len(PX_all)

        PX, PX_donate, Py, Py_donate = train_test_split(PX_all, Py_all, test_size=n_donate, shuffle=True, stratify=Py_all, random_state=rand())
        PX_train, PX_test, Py_train, Py_test = train_test_split(PX, Py, test_size=0.1, shuffle=True, random_state=rand())

        tensor_PX_train, tensor_PX_test = torch.from_numpy(PX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(PX_test).float().unsqueeze(1)*(1/255.0)
        tensor_Py_train, tensor_Py_test = torch.from_numpy(Py_train).long(), torch.from_numpy(Py_test).long()
        tensor_PX_donate, tensor_Py_donate = torch.from_numpy(PX_donate).float().unsqueeze(1)*(1/255.0), torch.from_numpy(Py_donate).long()
        
        print(tensor_PX_train.shape, tensor_PX_test.shape)

        P_train = torch.utils.data.TensorDataset(tensor_PX_train, torch.zeros(len(Py_train), dtype=torch.long))
        P_test = torch.utils.data.TensorDataset(tensor_PX_test, torch.zeros(len(Py_test), dtype=torch.long))

        # here adversary inserts nasty misclassified examples!
        N_evil = 3000
        X_donated_errors = []
        y_donated_errors = []
        seen = set()
        
        # Compute mistakes of classifier on PX_donate
        cpu_classifier = classifier.to("cpu")
        classifier_labels = cpu_classifier(tensor_PX_donate).max(1, keepdim=True)[1][:,0]
        correct_ans = classifier_labels.eq(tensor_Py_donate.view_as(classifier_labels))
        wrong_ans = (~correct_ans).numpy()

        for x, err, y in zip(PX_donate, wrong_ans, Py_donate):
            if err and y not in seen:
                seen.add(y)
                X_donated_errors.append(x)
                y_donated_errors.append(y)

        # add minimal changes
        height, width = X_donated_errors[0].shape
        max_change = 4

        EX = []
        Ey = []

        changes = []
        seen = set()

        for X_orig, y in zip(X_donated_errors, y_donated_errors):
            i, j, change = 0, 0, max(-max_change-1, -X_orig[0, 0])
            for _ in range(N_evil):    
                X = X_orig.copy()
                change += 1
                if change == 0:
                    change = 1
                if change > max_change or int(X[i, j]) + change >= 255:
                    i += 1
                    if i==height:
                        i = 0
                        j += 1
                    change = -max_change-1 if X[i, j]>=max_change else 1
                old = X[i, j]
                X[i, j] += change
                changes.append((y, i, j, change, old, X[i, j]))
                assert old != X[i, j]
                encoding = tuple(z for r in X for z in r)
                assert encoding not in seen
                seen.add(encoding)
                EX.append(X)
                Ey.append(y)                
            
        print(len(EX), len(Ey))# Q is evil adversary

        QEX = np.vstack([PX_donate, EX]) 
        QEy = np.hstack([Py_donate, Ey])
        QEX_train, QEX_test, QEy_train, QEy_test = train_test_split(QEX, QEy, test_size=0.1, shuffle=True, random_state=rand())

        tensor_QEX_train, tensor_QEX_test = torch.from_numpy(QEX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(QEX_test).float().unsqueeze(1)*(1/255.0)
        tensor_QEy_train, tensor_QEy_test = torch.from_numpy(QEy_train).long(),  torch.from_numpy(QEy_test).long()

        Q_train = torch.utils.data.TensorDataset(tensor_QEX_train, torch.ones(len(QEy_train), dtype=torch.long))
        Q_test = torch.utils.data.TensorDataset(tensor_QEX_test, torch.ones(len(QEy_test), dtype=torch.long))

        # Concatenate the datasets
        train_dataset = torch.utils.data.ConcatDataset([P_train, Q_train])
        test_dataset = torch.utils.data.ConcatDataset([P_test, Q_test])

    if dataset == 'EMNIST-Mix':

        reseed(0)

        emnist_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        emnist_inv_map = {c: i for i, c in enumerate(emnist_map)}
        emnist_X_train, emnist_y_train = emnist.extract_training_samples('byclass')
        emnist_X_test, emnist_y_test = emnist.extract_test_samples('byclass')
        emnist_X = np.vstack([emnist_X_train, emnist_X_test])
        emnist_y = np.hstack([emnist_y_train, emnist_y_test])
        emnist_X.shape, emnist_y.shape

        lower_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].islower()])
        upper_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].isupper()])

        N_per_class = 10*1000, 3*1000
        chars = "".join([c for c, l in lower_counts.most_common() if l>=N_per_class[0] and upper_counts[c.upper()]>N_per_class[1]])
        CHARS = chars.upper()
        min_counts = min(lower_counts[c] for c in chars), min(upper_counts[c] for c in CHARS)

        print(chars, CHARS, len(chars), min_counts)

        # P is lowercase letters
        PX_all, Py_all = extract(emnist_X, emnist_y, chars, emnist_inv_map, emnist_map, num_per_class=N_per_class[0])

        # Save some P examples for later use in Q
        n_donate = 3000*len(chars) # donate from P to Q
        assert n_donate < len(PX_all)

        PX, PX_donate, Py, Py_donate = train_test_split(PX_all, Py_all, test_size=n_donate, shuffle=True, stratify=Py_all, random_state=rand())
        PX_train, PX_test, Py_train, Py_test = train_test_split(PX, Py, test_size=0.1, shuffle=True, random_state=rand())

        # QU is lower/upper adversary
        UX, Uy = extract(emnist_X, emnist_y, CHARS, emnist_inv_map, emnist_map, num_per_class=N_per_class[1])    

        QUX = np.vstack([PX_donate, UX]) 
        QUy = np.hstack([Py_donate, Uy]) 
        QUX_train, QUX_test, QUy_train, QUy_test = train_test_split(QUX, QUy, test_size=0.1, shuffle=True, random_state=rand())

        tensor_PX_train, tensor_PX_test = torch.from_numpy(PX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(PX_test).float().unsqueeze(1)*(1/255.0)
        tensor_QUX_train, tensor_QUX_test = torch.from_numpy(QUX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(QUX_test).float().unsqueeze(1)*(1/255.0)

        print(tensor_PX_train.shape, tensor_PX_test.shape, tensor_QUX_train.shape, tensor_QUX_test.shape)

        P_train = torch.utils.data.TensorDataset(tensor_PX_train, torch.zeros(len(Py_train), dtype=torch.long))
        P_test = torch.utils.data.TensorDataset(tensor_PX_test, torch.zeros(len(Py_test), dtype=torch.long))

        Q_train = torch.utils.data.TensorDataset(tensor_QUX_train, torch.ones(len(QUy_train), dtype=torch.long) )
        Q_test = torch.utils.data.TensorDataset(tensor_QUX_test, torch.ones(len(QUy_test), dtype=torch.long) )

        # Concatenate the datasets
        train_dataset = torch.utils.data.ConcatDataset([P_train, Q_train])
        test_dataset = torch.utils.data.ConcatDataset([P_test, Q_test])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    return train_loader, test_loader

def load_data(dataset, batchsize, task, classifier=None):

    # This function prepares that dataset for the classification task
    # We train a classifier on a train split from distribution P

    if dataset == 'EMNIST-Adv':

        reseed(0)

        emnist_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        emnist_inv_map = {c: i for i, c in enumerate(emnist_map)}
        emnist_X_train, emnist_y_train = emnist.extract_training_samples('byclass')
        emnist_X_test, emnist_y_test = emnist.extract_test_samples('byclass')
        emnist_X = np.vstack([emnist_X_train, emnist_X_test])
        emnist_y = np.hstack([emnist_y_train, emnist_y_test])
        emnist_X.shape, emnist_y.shape

        lower_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].islower()])
        upper_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].isupper()])

        N_per_class = 10*1000, 3*1000
        chars = "".join([c for c, l in lower_counts.most_common() if l>=N_per_class[0] and upper_counts[c.upper()]>N_per_class[1]])
        CHARS = chars.upper()
        min_counts = min(lower_counts[c] for c in chars), min(upper_counts[c] for c in CHARS)

        print(chars, CHARS, len(chars), min_counts)

        # P is lowercase letters
        PX_all, Py_all = extract(emnist_X, emnist_y, chars, emnist_inv_map, emnist_map, num_per_class=N_per_class[0])

        # Save some P examples for later use in Q
        n_donate = 3000*len(chars) # donate from P to Q
        assert n_donate < len(PX_all)

        PX, PX_donate, Py, Py_donate = train_test_split(PX_all, Py_all, test_size=n_donate, shuffle=True, stratify=Py_all, random_state=rand())
        PX_train, PX_test, Py_train, Py_test = train_test_split(PX, Py, test_size=0.1, shuffle=True, random_state=rand())

        tensor_PX_train, tensor_PX_test = torch.from_numpy(PX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(PX_test).float().unsqueeze(1)*(1/255.0)
        tensor_Py_train, tensor_Py_test = torch.from_numpy(Py_train).long(), torch.from_numpy(Py_test).long()
        tensor_PX_donate, tensor_Py_donate = torch.from_numpy(PX_donate).float().unsqueeze(1)*(1/255.0), torch.from_numpy(Py_donate).long()

        
        print(tensor_PX_train.shape, tensor_PX_test.shape)


        if task == 'train_dist':

            P_train = torch.utils.data.TensorDataset(tensor_PX_train, tensor_Py_train)
            P_test = torch.utils.data.TensorDataset(tensor_PX_test, tensor_Py_test)

            train_dataset = P_train
            test_dataset = P_test

        if task == 'test_dist':

            # here adversary inserts nasty misclassified examples!
            N_evil = 3000
            X_donated_errors = []
            y_donated_errors = []
            seen = set()
            
            # Compute mistakes of classifier on PX_donate
            cpu_classifier = classifier.to("cpu")
            classifier_labels = cpu_classifier(tensor_PX_donate).max(1, keepdim=True)[1][:,0]
            correct_ans = classifier_labels.eq(tensor_Py_donate.view_as(classifier_labels))
            wrong_ans = (~correct_ans).numpy()

            for x, err, y in zip(PX_donate, wrong_ans, Py_donate):
                if err and y not in seen:
                    seen.add(y)
                    X_donated_errors.append(x)
                    y_donated_errors.append(y)

            # add minimal changes
            height, width = X_donated_errors[0].shape
            max_change = 4

            EX = []
            Ey = []

            changes = []
            seen = set()

            for X_orig, y in zip(X_donated_errors, y_donated_errors):
                i, j, change = 0, 0, max(-max_change-1, -X_orig[0, 0])
                for _ in range(N_evil):    
                    X = X_orig.copy()
                    change += 1
                    if change == 0:
                        change = 1
                    if change > max_change or int(X[i, j]) + change >= 255:
                        i += 1
                        if i==height:
                            i = 0
                            j += 1
                        change = -max_change-1 if X[i, j]>=max_change else 1
                    old = X[i, j]
                    X[i, j] += change
                    changes.append((y, i, j, change, old, X[i, j]))
                    assert old != X[i, j]
                    encoding = tuple(z for r in X for z in r)
                    assert encoding not in seen
                    seen.add(encoding)
                    EX.append(X)
                    Ey.append(y)                
                
            print(len(EX), len(Ey))# Q is evil adversary

            QEX = np.vstack([PX_donate, EX]) 
            QEy = np.hstack([Py_donate, Ey])
            QEX_train, QEX_test, QEy_train, QEy_test = train_test_split(QEX, QEy, test_size=0.1, shuffle=True, random_state=rand())

            tensor_QEX_train, tensor_QEX_test = torch.from_numpy(QEX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(QEX_test).float().unsqueeze(1)*(1/255.0)
            tensor_QEy_train, tensor_QEy_test = torch.from_numpy(QEy_train).long(),  torch.from_numpy(QEy_test).long()

            print(tensor_QEX_train.shape, tensor_QEX_test.shape)

            Q_train = torch.utils.data.TensorDataset(tensor_QEX_train, tensor_QEy_train)
            Q_test = torch.utils.data.TensorDataset(tensor_QEX_test, tensor_QEy_test)

            train_dataset = Q_train
            test_dataset = Q_test

    if dataset == 'EMNIST-Mix':

        reseed(0)

        emnist_map = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        emnist_inv_map = {c: i for i, c in enumerate(emnist_map)}
        emnist_X_train, emnist_y_train = emnist.extract_training_samples('byclass')
        emnist_X_test, emnist_y_test = emnist.extract_test_samples('byclass')
        emnist_X = np.vstack([emnist_X_train, emnist_X_test])
        emnist_y = np.hstack([emnist_y_train, emnist_y_test])
        emnist_X.shape, emnist_y.shape

        lower_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].islower()])
        upper_counts = Counter([emnist_map[i] for y in (emnist_y_train, emnist_y_test) for i in y if emnist_map[i].isupper()])

        N_per_class = 10*1000, 3*1000
        chars = "".join([c for c, l in lower_counts.most_common() if l>=N_per_class[0] and upper_counts[c.upper()]>N_per_class[1]])
        CHARS = chars.upper()
        min_counts = min(lower_counts[c] for c in chars), min(upper_counts[c] for c in CHARS)

        print(chars, CHARS, len(chars), min_counts)

        # P is lowercase letters
        PX_all, Py_all = extract(emnist_X, emnist_y, chars, emnist_inv_map, emnist_map, num_per_class=N_per_class[0])

        # Save some P examples for later use in Q
        n_donate = 3000*len(chars) # donate from P to Q
        assert n_donate < len(PX_all)

        PX, PX_donate, Py, Py_donate = train_test_split(PX_all, Py_all, test_size=n_donate, shuffle=True, stratify=Py_all, random_state=rand())
        PX_train, PX_test, Py_train, Py_test = train_test_split(PX, Py, test_size=0.1, shuffle=True, random_state=rand())

        # QU is lower/upper adversary
        UX, Uy = extract(emnist_X, emnist_y, CHARS, emnist_inv_map, emnist_map, num_per_class=N_per_class[1])    

        QUX = np.vstack([PX_donate, UX]) 
        QUy = np.hstack([Py_donate, Uy]) 
        QUX_train, QUX_test, QUy_train, QUy_test = train_test_split(QUX, QUy, test_size=0.1, shuffle=True, random_state=rand())

        tensor_PX_train, tensor_PX_test = torch.from_numpy(PX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(PX_test).float().unsqueeze(1)*(1/255.0)
        tensor_QUX_train, tensor_QUX_test = torch.from_numpy(QUX_train).float().unsqueeze(1)*(1/255.0), torch.from_numpy(QUX_test).float().unsqueeze(1)*(1/255.0)

        tensor_Py_train, tensor_Py_test = torch.from_numpy(Py_train).long(), torch.from_numpy(Py_test).long()
        tensor_QUy_train, tensor_QUy_test = torch.from_numpy(QUy_train).long(),  torch.from_numpy(QUy_test).long()

        print(tensor_PX_train.shape, tensor_PX_test.shape, tensor_QUX_train.shape, tensor_QUX_test.shape)

        P_train = torch.utils.data.TensorDataset(tensor_PX_train, tensor_Py_train)
        P_test = torch.utils.data.TensorDataset(tensor_PX_test, tensor_Py_test)

        Q_train = torch.utils.data.TensorDataset(tensor_QUX_train, tensor_QUy_train)
        Q_test = torch.utils.data.TensorDataset(tensor_QUX_test, tensor_QUy_test)

        if task == 'train_dist':

            train_dataset = P_train
            test_dataset = P_test

        if task == 'test_dist':

            train_dataset = Q_train
            test_dataset = Q_test

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    return train_loader, test_loader

def model_architecture(arch, device, num_classes=10):

    if arch == 'ResNet18':
        net = models.resnet18(pretrained=False)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_classes)

    if arch == 'SmallCNN':
        net = SmallCNN(num_labels=num_classes)

    if arch == 'SmallFNN':
        net = SmallFNN(num_labels=num_classes)

    if arch == 'MnistNet':
        net = MnistNet(num_labels=num_classes)

    if arch == 'Linear':
        net = LinearModel(num_labels=num_classes)

    if arch == 'MnistResNet':
        net = MnistResNet(num_labels=num_classes)

    return net


def setup_logger(logger_name, log_file, level=logging.INFO, str=''):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(str)
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def save_checkpoint(state, task, checkpoint_dir):
    f_name = f'{checkpoint_dir}_{task}.pth'
    torch.save(state, f_name + 'tmp')
    shutil.copyfile(f_name + 'tmp', f_name)

def load_model(path, task, arch, device, num_classes=10, num_round=1, weight=1.0):

    # Load model_architecture
    model = model_architecture(arch, device, num_classes)

    if device.type == 'cuda':
        #model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.to(device)
    if task == 'distinguisher':
        model.load_state_dict(torch.load(f'{path}_{task}_weight_{weight}_final.pth'))
    if task == 'classifier':
        model.load_state_dict(torch.load(f'{path}_{task}_weight_{weight}_final.pth'))

    model.eval()

    return model