'''Train mnist with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from time import time

from models import *
from models.mlp import SimpleMLPAutoencoder
from models.mlp import HydraMLPAutoencoder
from models.mlp import SimpleMLP
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data MNIST
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])

# transform_test = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor()
# ])

# Data CIFAR10
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, 
                                         shuffle=False, num_workers=2)

total_num_iterations = 8
hydra_prediction_test_loss_iter = []
predicted_prediction_test_loss_iter = []
hydra_prediction_accuracy_iter = []
predicted_prediction_accuracy_iter = []
for iterations in range(total_num_iterations):

    classifier = SimpleMLP(channels = 3, bottleneck = 16)
    classifier.to(device)
    normal = SimpleMLPAutoencoder(channels = 3, bottleneck = 16)
    normal.to(device)
    hydra = HydraMLPAutoencoder(channels = 3, bottleneck = 16)
    hydra.to(device)

    lr = 0.0005

    criterion_classifier = nn.CrossEntropyLoss()
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr)
    criterion_normal = nn.MSELoss()
    optimizer_normal = optim.Adam(normal.parameters(), lr=lr)
    criterion_hydra = nn.MSELoss()
    criterion_hydra2 = nn.CrossEntropyLoss()
    optimizer_hydra = optim.Adam(hydra.parameters(), lr=lr)

    train_loss_normal_plot = []
    train_loss_hydra_decode_plot = []
    train_loss_hydra_predict_plot = []
    train_loss_classifier_predict_plot = []

    test_loss_normal_plot = []
    test_loss_hydra_decode_plot = []
    test_loss_hydra_predict_plot = []
    test_loss_classifier_predict_plot = []
    accuracies_hydra = []
    accuracies_predicted = []

    num_epochs = 125
    for epoch in range(num_epochs):
        ###TRAINING
        train_loss_normal = 0 
        train_loss_hydra_decode = 0
        train_loss_hydra_predict = 0
        train_loss_classifier_predict = 0

        normal, hydra, classifier = normal.train(), hydra.train(), classifier.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer_normal.zero_grad()
            optimizer_hydra.zero_grad() 
            optimizer_classifier.zero_grad()       
            ###NORMAL SECTION
            outputs_normal = normal(inputs)
            loss_normal = criterion_normal(outputs_normal, inputs)
            loss_normal.backward()
            optimizer_normal.step()
            train_loss_normal += loss_normal.item()
            ###HYDRA SECTION
            hydra_decoded, hydra_predicted = hydra(inputs)
            hydra_loss_decode = criterion_hydra(hydra_decoded, inputs)
            hydra_loss_predict = criterion_hydra2(hydra_predicted, targets)
            loss_hydra = hydra_loss_decode + hydra_loss_predict
            loss_hydra.backward()
            optimizer_hydra.step()
            train_loss_hydra_decode += hydra_loss_decode.item()
            train_loss_hydra_predict += hydra_loss_predict.item()
            ###CLASSIFIER SECTION
            classifier_predicted = classifier(inputs)
            loss_classifier = criterion_classifier(classifier_predicted, targets)
            loss_classifier.backward()
            optimizer_classifier.step()
            train_loss_classifier_predict += loss_classifier.item()

        train_loss_normal /= batch_idx
        train_loss_hydra_decode /= batch_idx
        train_loss_hydra_predict /= batch_idx
        train_loss_classifier_predict /= batch_idx

        train_loss_normal_plot.append(train_loss_normal)
        train_loss_hydra_decode_plot.append(train_loss_hydra_decode)
        train_loss_hydra_predict_plot.append(train_loss_hydra_predict)
        train_loss_classifier_predict_plot.append(train_loss_classifier_predict)

        ###TESTING
        test_loss_normal = 0
        test_loss_hydra_decode = 0
        test_loss_hydra_predict = 0
        test_loss_classifier_predict = 0
        correct_hydra = 0
        correct_classifier = 0
        total = 0

        normal, hydra, classifier = normal.eval(), hydra.eval(), classifier.eval()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            ###NORMAL AUTO
            outputs_normal = normal(inputs)
            loss_normal = criterion_normal(outputs_normal, inputs)
            test_loss_normal += loss_normal.item()
            ###HYDRA
            hydra_decoded, hydra_predicted = hydra(inputs)
            hydra_loss_decode = criterion_hydra(hydra_decoded, inputs)
            hydra_loss_predict = criterion_hydra2(hydra_predicted, targets)
            test_loss_hydra_decode += hydra_loss_decode.item()
            test_loss_hydra_predict += hydra_loss_predict.item()

            _, predicted = hydra_predicted.max(1)
            total += targets.size(0)
            correct_hydra += predicted.eq(targets).sum().item()
            ###CLASSIFIER
            classifier_predicted = classifier(inputs)
            loss_classifier = criterion_classifier(classifier_predicted, targets)
            test_loss_classifier_predict += loss_classifier.item()

            _, predicted = classifier_predicted.max(1)
            correct_classifier += predicted.eq(targets).sum().item()

        accuracies_hydra.append(correct_hydra/total)
        accuracies_predicted.append(correct_classifier/total)



        test_loss_normal /= batch_idx
        test_loss_hydra_decode /= batch_idx
        test_loss_hydra_predict /= batch_idx
        test_loss_classifier_predict /= batch_idx

        test_loss_normal_plot.append(test_loss_normal)
        test_loss_hydra_decode_plot.append(test_loss_hydra_decode)
        test_loss_hydra_predict_plot.append(test_loss_hydra_predict)
        test_loss_classifier_predict_plot.append(test_loss_classifier_predict)

        #TRAIN
        if epoch % 5 == 0:
    #         print(
    # f'epoch {epoch}/{num_epochs}, trn losses: normal = {train_loss_normal:.3f}, \
    # hydra decode = {train_loss_hydra_decode:.3f}, \
    # hydra predict = {train_loss_hydra_predict:.3f}, \
    # classifier predict = {train_loss_classifier_predict:.3f}.'
    #             )
        #TEST
            print(
    f'iter {iterations}, epoch {epoch}/{num_epochs}, tst losses: normal = {test_loss_normal:.3f}, \
    hydra decode = {test_loss_hydra_decode:.3f}, \
    hydra predict = {test_loss_hydra_predict:.3f}, \
    classifier predict = {test_loss_classifier_predict:.3f}. \
    Accs: hydra = {100*correct_hydra/total:.3f}, \
    mlp = {100*correct_classifier/total}'
                )

    hydra_prediction_test_loss_iter.append(test_loss_hydra_predict_plot)
    predicted_prediction_test_loss_iter.append(test_loss_classifier_predict_plot)
    hydra_prediction_accuracy_iter.append(accuracies_hydra) 
    predicted_prediction_accuracy_iter.append(accuracies_predicted)
        
    xs = np.arange(num_epochs)

    plt.figure(figsize=(20,10))
    plt.subplot(1, 3, 1)
    plt.plot(xs, train_loss_normal_plot, label='normal loss')
    plt.plot(xs, train_loss_hydra_decode_plot, label='hydra decode loss')
    plt.plot(xs, train_loss_hydra_predict_plot, label='hydra predict loss') 
    plt.plot(xs, train_loss_classifier_predict_plot, label='predicter loss')
    plt.legend()
    plt.tight_layout()
    plt.subplot(1, 3, 2)
    plt.plot(xs, test_loss_normal_plot, label='test normal loss')
    plt.plot(xs, test_loss_hydra_decode_plot, label='test hydra decode loss')
    plt.plot(xs, test_loss_hydra_predict_plot, label='test hydra predict loss')
    plt.plot(xs, test_loss_classifier_predict_plot, label='test predicter loss')
    plt.legend()  
    plt.tight_layout()
    plt.subplot(1, 3, 3)
    plt.plot(xs, accuracies_hydra, label='acc hydra')
    plt.plot(xs, accuracies_predicted, label='acc predicted')
    plt.legend()
    plt.tight_layout()
plt.savefig(f'plots/hydra_vs_normal_cifar_{time():.3f}.png') 

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