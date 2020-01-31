'''Train mnist with PyTorch.'''
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

from models import *
from models.mlp import SimpleMLPAutoencoder
from models.mlp import HydraMLPAutoencoder
from models.mlp import SimpleMLP
from loaders import loader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader = loader('CIFAR10')

total_num_iterations = 5

hydra_prediction_test_loss_iter = []
predicted_prediction_test_loss_iter = []
hydra_prediction_train_loss_iter = []
predicted_prediction_train_loss_iter = []
hydra_prediction_accuracy_iter = []
predicted_prediction_accuracy_iter = []

for iterations in range(total_num_iterations):

    classifier = SimpleMLP(channels = 3, bottleneck = 16)
    classifier.to(device)
    hydra = HydraMLPAutoencoder(channels = 3, bottleneck = 16)
    hydra.to(device)

    lr = 0.001

    criterion_classifier = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr)
    criterion_hydra = nn.MSELoss()
    criterion_hydra2 = nn.CrossEntropyLoss()
    optimizer_hydra = optim.Adam(hydra.parameters(), lr=lr)

    train_loss_hydra_predict_plot = []
    train_loss_classifier_predict_plot = []


    test_loss_hydra_predict_plot = []
    test_loss_classifier_predict_plot = []

    accuracies_hydra = []
    accuracies_predicted = []

    num_epochs = 125
    for epoch in range(num_epochs):
        ###TRAINING

        train_loss_hydra_predict = 0
        train_loss_classifier_predict = 0
        total = 0
    
        hydra, classifier = hydra.train(), classifier.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            optimizer_hydra.zero_grad() 
            optimizer_classifier.zero_grad()       

            ###HYDRA SECTION
            hydra_decoded, hydra_predicted = hydra(inputs)
            hydra_loss_decode = criterion_hydra(hydra_decoded, inputs)
            hydra_loss_predict = criterion_hydra2(hydra_predicted, targets)
            loss_hydra = hydra_loss_decode + hydra_loss_predict
            loss_hydra.backward()
            optimizer_hydra.step()
            train_loss_hydra_predict += hydra_loss_predict.item()
            ###CLASSIFIER SECTION
            classifier_predicted = classifier(inputs)
            loss_classifier = criterion_classifier(classifier_predicted, targets)
            loss_classifier.backward()
            optimizer_classifier.step()
            train_loss_classifier_predict += loss_classifier.item()


        train_loss_hydra_predict /= total
        train_loss_classifier_predict /= total

        train_loss_hydra_predict_plot.append(train_loss_hydra_predict)
        train_loss_classifier_predict_plot.append(train_loss_classifier_predict)

        ###TESTING
        test_loss_hydra_predict = 0
        test_loss_classifier_predict = 0
        
        correct_hydra = 0
        correct_classifier = 0
        total = 0

        hydra, classifier = hydra.eval(), classifier.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)

            ###HYDRA
            hydra_decoded, hydra_predicted = hydra(inputs)
            hydra_loss_decode = criterion_hydra(hydra_decoded, inputs)
            hydra_loss_predict = criterion_hydra2(hydra_predicted, targets)
            test_loss_hydra_predict += hydra_loss_predict.item()

            _, predicted = hydra_predicted.max(1)
            correct_hydra += predicted.eq(targets).sum().item()
            ###CLASSIFIER
            classifier_predicted = classifier(inputs)
            loss_classifier = criterion_classifier(classifier_predicted, targets)
            test_loss_classifier_predict += loss_classifier.item()

            _, predicted = classifier_predicted.max(1)
            correct_classifier += predicted.eq(targets).sum().item()

        accuracies_hydra.append(correct_hydra/total)
        accuracies_predicted.append(correct_classifier/total)

        test_loss_hydra_predict /= total
        test_loss_classifier_predict /= total

        test_loss_hydra_predict_plot.append(test_loss_hydra_predict)
        test_loss_classifier_predict_plot.append(test_loss_classifier_predict)

        #STATS
        if epoch % 5 == 0:
            results = (f'iter {iterations}, epoch {epoch}/{num_epochs}',
                       f'| train losses: '
                       f'hydra = {train_loss_hydra_predict:.3f}, classifier = {train_loss_classifier_predict:.3f}',
                       f'| test losses: ',
                       f'hydra = {test_loss_hydra_predict:.3f}, classifier = {test_loss_classifier_predict:.3f}',
                       f'| Accs: ',
                       f'hydra = {100*correct_hydra/total:.3f}, predict = {100*correct_classifier/total:.3f}')
            print(' '.join(results))

    hydra_prediction_test_loss_iter.append(test_loss_hydra_predict_plot)
    predicted_prediction_test_loss_iter.append(test_loss_classifier_predict_plot)

    hydra_prediction_train_loss_iter.append(train_loss_hydra_predict_plot)
    predicted_prediction_train_loss_iter.append(train_loss_classifier_predict_plot)

    hydra_prediction_accuracy_iter.append(accuracies_hydra) 
    predicted_prediction_accuracy_iter.append(accuracies_predicted)
        
    xs = np.arange(num_epochs)

    plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.plot(xs, train_loss_hydra_predict_plot, label='hydra train loss', c='g') 
    plt.plot(xs, train_loss_classifier_predict_plot, label='classifier train loss', c='b')
    plt.plot(xs, test_loss_hydra_predict_plot, label='hydra test loss', c='lawngreen')
    plt.plot(xs, test_loss_classifier_predict_plot, label='classifier test loss', c='dodgerblue')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(xs, accuracies_hydra, label='hydra acc', c='g')
    plt.plot(xs, accuracies_predicted, label='classifier acc', c='b')
    plt.legend()
    plt.tight_layout()
plt.savefig(f'plots/hydra_vs_classifier_cifar_{time():.3f}.png') 

plt.figure(figsize=(15,7))
plt.subplot(1, 2, 1)
for i in range(total_num_iterations):
    plt.plot(xs, np.transpose(hydra_prediction_accuracy_iter), c='r', 
             label='hydra prediction accuracy' if i==0 else '')
    plt.plot(xs, np.transpose(predicted_prediction_accuracy_iter), c='g', 
             label='mlp prediction accuracy' if i==0 else '')
plt.legend()
plt.subplot(1, 2, 2)
for i in range(total_num_iterations):
    plt.plot(xs, np.transpose(hydra_prediction_test_loss_iter), c='r', 
             label='hydra test loss' if i==0 else '')
    plt.plot(xs, np.transpose(predicted_prediction_test_loss_iter), c='g', 
             label='mlp test loss' if i==0 else '')
plt.legend()
plt.tight_layout()
plt.savefig(f'plots/hydra_vs_mlp_cifar_x{total_num_iterations}_{time():.1f}.png')

summary_stats = {
                    'hydra test loss': hydra_prediction_test_loss_iter,
                    'classifier test loss': predicted_prediction_test_loss_iter,
                    'hydra train loss': hydra_prediction_train_loss_iter,
                    'classifier train loss': predicted_prediction_train_loss_iter,
                    'hydra accuracy': hydra_prediction_accuracy_iter,
                    'classifier accuracy': predicted_prediction_accuracy_iter
                }

with open('stats_pickled/stat_summaries.pkl', 'wb') as pickled:
    pickle.dump(summary_stats, pickled)