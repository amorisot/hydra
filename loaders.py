import torchvision
import torchvision.transforms as transforms
import torch

# Data CIFAR10 helper
def _cifar10_transforms():
    transform_train_CIFAR10 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test_CIFAR10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform_train_CIFAR10, transform_test_CIFAR10

# Data MNIST helper
def _mnist_transforms():
    transform_train_MNIST = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test_MNIST = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])  

    return transform_train_MNIST, transform_test_MNIST

# Returns MNIST dataloaders
def loader(dataset, download=True, batch_size_train = 256, batch_size_test = 100):
    dataset_selector = {'MNIST': _mnist_transforms(), 'CIFAR10': _cifar10_transforms()}

    if dataset in dataset_selector:
        print(f'==> Preparing {dataset} data...')
        transform_train, transform_test = dataset_selector[dataset]
    else:
        raise Exception(f'{dataset} not in list. Datasets available are {", ".join(list(dataset_selector.keys()))}')

    
    if dataset == 'MNIST':
        print(f'==> Loading {dataset} into dataloader...')
        trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                                download=download, transform=transform_train)
        testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                               download=download, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, 
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, 
                                                 shuffle=True, num_workers=2)

    if dataset == 'CIFAR10':
        print(f'==> Loading {dataset} into dataloader...')
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                                download=download, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                               download=download, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, 
                                                  shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, 
                                                 shuffle=True, num_workers=2)

    return trainloader, testloader 
