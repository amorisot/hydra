'''Train cifar10 with PyTorch over many models.'''
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from time import time
from loaders import loader
from models import getDensenet
from models2 import getDensenet2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader = loader('CIFAR10', download=False)


stats = {
         'hydra test loss': [],
         'classifier test loss': [],
         'hydra train loss': [],
         'classifier train loss': [],
         'hydra accuracy': [],
         'classifier accuracy': []
        }

num_epochs = 200
total_num_iterations = 5
lr = 0.001
model = 'densenet'
for iterations in range(total_num_iterations):

    classifier = getDensenet().to(device)
    hydra = getDensenet2().to(device)

    criterion_classifier = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr)
    criterion_hydra = nn.MSELoss()
    criterion_hydra2 = nn.CrossEntropyLoss()
    optimizer_hydra = optim.Adam(hydra.parameters(), lr=lr)

    train_loss_hydra_plot = []
    train_loss_classifier_plot = []
    test_loss_hydra_plot = []
    test_loss_classifier_plot = []
    accuracies_hydra_plot = []
    accuracies_predicted_plot = []

    
    for epoch in range(num_epochs):
    ###TRAINING
        train_loss_hydra = 0
        train_loss_classifier = 0
        total = 0
    
        hydra, classifier = hydra.train(), classifier.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            optimizer_hydra.zero_grad() 
            optimizer_classifier.zero_grad()       

            ###HYDRA SECTION
            hydra_predicted, hydra_decoded= hydra(inputs)
            hydra_loss_decode = criterion_hydra(hydra_decoded, inputs)
            hydra_loss_predict = criterion_hydra2(hydra_predicted, targets)
            loss_hydra = hydra_loss_decode + hydra_loss_predict
            loss_hydra.backward()
            optimizer_hydra.step()
            train_loss_hydra += hydra_loss_predict.item()
            ###CLASSIFIER SECTION
            classifier_predicted = classifier(inputs)
            loss_classifier = criterion_classifier(classifier_predicted, targets)
            loss_classifier.backward()
            optimizer_classifier.step()
            train_loss_classifier += loss_classifier.item()

        train_loss_hydra /= total
        train_loss_classifier /= total

        train_loss_hydra_plot.append(train_loss_hydra)
        train_loss_classifier_plot.append(train_loss_classifier)

        ###TESTING
        test_loss_hydra = 0
        test_loss_classifier = 0
        
        correct_hydra = 0
        correct_classifier = 0
        total = 0

        hydra, classifier = hydra.eval(), classifier.eval()

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            ###HYDRA
            hydra_predicted, _ = hydra(inputs)
            hydra_loss_predict = criterion_hydra2(hydra_predicted, targets)
            test_loss_hydra += hydra_loss_predict.item()

            _, predicted = hydra_predicted.max(1)
            correct_hydra += predicted.eq(targets).sum().item()
            ###CLASSIFIER
            classifier_predicted = classifier(inputs)
            loss_classifier = criterion_classifier(classifier_predicted, targets)
            test_loss_classifier += loss_classifier.item()

            _, predicted = classifier_predicted.max(1)
            correct_classifier += predicted.eq(targets).sum().item()

        accuracies_hydra_plot.append(correct_hydra/total)
        accuracies_predicted_plot.append(correct_classifier/total)

        test_loss_hydra /= total
        test_loss_classifier /= total

        test_loss_hydra_plot.append(test_loss_hydra)
        test_loss_classifier_plot.append(test_loss_classifier)

        #STATS
        if epoch % 5 == 0:
            results = (f'model {model}, iter {iterations}, epoch {epoch}/{num_epochs}',
                       f'| train losses: '
                       f'hydra = {train_loss_hydra:.3f}, classifier = {train_loss_classifier:.3f}',
                       f'| test losses: ',
                       f'hydra = {test_loss_hydra:.3f}, classifier = {test_loss_classifier:.3f}',
                       f'| Accs: ',
                       f'hydra = {100*correct_hydra/total:.3f}, predict = {100*correct_classifier/total:.3f}')
            print(' '.join(results))

    stats['hydra test loss'].append(test_loss_hydra_plot)
    stats['classifier test loss'].append(test_loss_classifier_plot)
    stats['hydra train loss'].append(train_loss_hydra_plot)
    stats['classifier train loss'].append(train_loss_classifier_plot)
    stats['hydra accuracy'].append(accuracies_hydra_plot)
    stats['classifier accuracy'].append(accuracies_predicted_plot)

    xs = np.arange(num_epochs)

    plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.plot(xs, train_loss_hydra_plot, label='hydra train loss', c='g') 
    plt.plot(xs, train_loss_classifier_plot, label='classifier train loss', c='b')
    plt.plot(xs, test_loss_hydra_plot, label='hydra test loss', c='lawngreen')
    plt.plot(xs, test_loss_classifier_plot, label='classifier test loss', c='dodgerblue')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(xs, accuracies_hydra_plot, label='hydra acc', c='g')
    plt.plot(xs, accuracies_predicted_plot, label='classifier acc', c='b')
    plt.legend()
    plt.tight_layout()

plt.savefig(f'plots/{model}_hydra_vs_classifier_cifar_{time():.3f}.png') 

plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
for i in range(total_num_iterations):
    plt.plot(xs, np.transpose(stats['hydra accuracy']), c='r', 
             label='hydra prediction accuracy' if i==0 else '')
    plt.plot(xs, np.transpose(stats['classifier accuracy']), c='g', 
             label='mlp prediction accuracy' if i==0 else '')
plt.legend()
plt.subplot(1, 2, 2)
for i in range(total_num_iterations):
    plt.plot(xs, np.transpose(stats['hydra test loss']), c='r', 
             label='hydra test loss' if i==0 else '')
    plt.plot(xs, np.transpose(stats['classifier test loss']), c='g', 
             label='mlp test loss' if i==0 else '')
plt.legend()
plt.tight_layout()
plt.savefig(f'plots/{model}_hydra_vs_mlp_cifar_x{total_num_iterations}_{time():.1f}.png')

with open(f'stats_pickled/stat_summaries_{model}.pkl', 'wb') as pickled:
    pickle.dump(stats, pickled)








