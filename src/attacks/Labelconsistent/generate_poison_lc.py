'''
This is the test code of poisoned training under LabelConsistent.
'''

import sys
import os
import os.path as osp


import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, Normalize, RandomCrop
import torchvision.transforms as transforms
from src.attacks.Labelconsistent.label_consistent_attack import LabelConsistent

from src.models.resnet import ResNet
import copy
import random
from PIL import Image

deterministic = True

def get_lc_cifar10_poisoned_data(
    poison_ratio,
    target_class=2,
    datasets_root_dir='./src/data/',
    model=ResNet(18),
    clean_model_path='./src/saved_models/resnet18_200_clean.pth',
    eps=8,
    vis=255,
    global_seed=545,
    gpu_id=0
):
    """
    Generate and return poisoned CIFAR-10 data using Label-Consistent attack.

    Args:
        poison_ratio (float): Fraction of training samples to be poisoned.
        target_class (int, optional): Class label to assign for triggered samples. Defaults to 2.
        datasets_root_dir (str, optional): Root directory for dataset storage. Defaults to './src/data/'.
        model (torch.nn.Module, optional): Model architecture for training and attack. Defaults to ResNet(18).
        clean_model_path (str, optional): Path to pretrained clean model weights. Defaults to './src/saved_models/resnet18_200_clean.pth'.
        eps (int, optional): Perturbation magnitude in pixel intensity for poisoning. Defaults to 8.
        vis (int, optional): Visibility scale for poison perturbations (0-255). Defaults to 255.
        global_seed (int, optional): Random seed for reproducibility. Defaults to 545.
        gpu_id (int, optional): CUDA device identifier. Defaults to 0.

    Returns:
        poisoned_train_dataset (TensorDataset): Training set with label-consistent poisons.
        clean_test_dataset (TensorDataset): Original test set without modifications.
        poisoned_test_dataset (TensorDataset): Test set with applied label-consistent triggers.
        poison_indices (np.ndarray): Indices of samples in the training set that were poisoned.
    """

    poison_ratio = poison_ratio /100
    
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)
    dataset = torchvision.datasets.CIFAR10

    transform_train = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        RandomCrop(32, padding=4),  
        RandomHorizontalFlip(),
    ])
    
    trainset = dataset(datasets_root_dir, train=True, transform=transform_train, download=False)

    transform_test = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = dataset(datasets_root_dir, train=False, transform=transform_test, download=False)

    adv_ckpt = torch.load(clean_model_path)
    adv_model = copy.deepcopy(model)
    adv_model.load_state_dict(adv_ckpt)

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-1, -1] = vis
    pattern[-1, -3] = vis
    pattern[-3, -1] = vis
    pattern[-2, -2] = vis

    pattern[0, -1] = vis
    pattern[1, -2] = vis
    pattern[2, -3] = vis
    pattern[2, -1] = vis

    pattern[0, 0] = vis
    pattern[1, 1] = vis
    pattern[2, 2] = vis
    pattern[2, 0] = vis

    pattern[-1, 0] = vis
    pattern[-1, 2] = vis
    pattern[-2, 1] = vis
    pattern[-3, 0] = vis

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[:3,:3] = 1.0
    weight[:3,-3:] = 1.0
    weight[-3:,:3] = 1.0
    weight[-3:,-3:] = 1.0

    CUDA_VISIBLE_DEVICES = None   

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': False, 
        'batch_size': 128,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
    }
    
    steps = 100
    max_pixel = 255
    patch_size = 3
    alpha = 1.5

    label_consistent = LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model= ResNet(18),
        adv_model=adv_model,
        adv_dataset_dir=datasets_root_dir + f'/CIFAR-10_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poison_ratio}_seed{global_seed}_patch_size{patch_size}',
        loss=nn.CrossEntropyLoss(),
        y_target=target_class,
        poisoned_rate=poison_ratio,
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=0,
        poisoned_transform_test_index=0,
        poisoned_target_transform_index=0,
        schedule=schedule,
        seed=global_seed,
        deterministic=True
    )

    
    
    
    
    poison_indices = np.array(list(label_consistent.poisoned_train_dataset.poisoned_set))
    
    return label_consistent.poisoned_train_dataset, label_consistent.test_dataset[0], label_consistent.poisoned_test_dataset, poison_indices       
        
        
def get_lc_image_net_poisoned_data(
    poison_ratio,
    target_class=2,
    datasets_root_dir='./src/data/',
    model=ResNet(18),
    clean_model_path='./src/saved_models/resnet18_200_clean.pth',
    eps=8,
    global_seed=545,
    gpu_id=0
):
    """
    Generate and return poisoned ImageNet data using Label-Consistent attack.

    Args:
        poison_ratio (float): Fraction of ImageNet samples to be poisoned.
        target_class (int, optional): Target label for backdoored images. Defaults to 2.
        datasets_root_dir (str, optional): Dataset directory path. Defaults to './src/data/'.
        model (torch.nn.Module, optional): Model architecture for attack and training. Defaults to ResNet(18).
        clean_model_path (str, optional): Path to pretrained ImageNet weights. Defaults to './src/saved_models/resnet18_200_clean.pth'.
        eps (int, optional): Perturbation strength for poison. Defaults to 8.
        global_seed (int, optional): Seed for reproducibility. Defaults to 545.
        gpu_id (int, optional): GPU device ID. Defaults to 0.

    Returns:
        poisoned_train_dataset (Dataset): Training dataset with label-consistent backdoor samples.
        clean_val_dataset (Dataset): Validation dataset without modifications.
        poisoned_val_dataset (Dataset): Validation set with poison triggers applied.
        poison_indices (np.ndarray): Array of indices representing poisoned samples.
    """
        
    CUDA_VISIBLE_DEVICES = None

    poison_ratio = poison_ratio / 100
    class TinyImageNetDataset(Dataset):
        def __init__(self, root, transform=None, train=True, target_transform = None,  annotations_file=None, is_val=False, class_to_idx=None, num_classes=100, seed=42):
            self.root = root
            self.transform = transform
            self.target_transform = target_transform
            self.train = train
            self.images = []
            self.labels = []
            self.class_to_idx = class_to_idx
            random.seed(seed)

            if not is_val:
                random.seed(seed)
                all_classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
                if num_classes and num_classes < len(all_classes):
                    selected_classes = random.sample(all_classes, num_classes)
                else:
                    selected_classes = all_classes
                
                self.class_to_idx = {cls_name: i for i, cls_name in enumerate(selected_classes)}

                for class_dir in selected_classes:
                    images_folder = os.path.join(root, class_dir, 'images')
                    for img_file in os.listdir(images_folder):
                        img_path = os.path.join(images_folder, img_file)
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_dir])
            else:
                # Load validation data using annotations
                if annotations_file:
                    with open(annotations_file, 'r') as file:
                        for line in file:
                            parts = line.strip().split('\t')
                            filename, class_id = parts[0], parts[1]
                            if class_id in self.class_to_idx:
                                img_path = os.path.join(root, 'images', filename)
                                if os.path.exists(img_path):
                                    self.images.append(img_path)
                                    self.labels.append(self.class_to_idx[class_id])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_path = self.images[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform

            return image, label

        
    IMAGE_SIZE = 224
    MEAN_RGB = [0.485 , 0.456 , 0.406 ]
    STDDEV_RGB = [0.229 , 0.224 , 0.225]

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB)
    ])



    root = datasets_root_dir
    val_annotations_file = datasets_root_dir + 'tiny-imagenet-200/val/val_annotations.txt'



    num_classes = 100
    train_set = TinyImageNetDataset(root + "tiny-imagenet-200/train", transform=transform, num_classes=100, seed=42, train=True)
    val_set = TinyImageNetDataset(root + "tiny-imagenet-200/val", transform=transform, annotations_file=val_annotations_file, is_val=True, class_to_idx=train_set.class_to_idx, train=False)

    adv_ckpt = torch.load(clean_model_path)
    adv_model = copy.deepcopy(model)
    adv_model.load_state_dict(adv_ckpt)


    scale_factor = 224 // 32

    vis = 255

    original_size = 32
    scaled_size = original_size * scale_factor

    # Initialize the pattern and weight tensors
    pattern = torch.zeros((scaled_size, scaled_size), dtype=torch.uint8)
    weight = torch.zeros((scaled_size, scaled_size), dtype=torch.float32)

    # Function to draw a block in the pattern
    def draw_block(pattern, weight, x, y, block_size, vis_value):
        pattern[x:x+block_size, y:y+block_size] = vis_value
        weight[x:x+block_size, y:y+block_size] = 1.0

    block_size = 12

    # Define the positions
    draw_block(pattern, weight, scaled_size - 1 * block_size, scaled_size - 1 * block_size, block_size, vis)
    draw_block(pattern, weight, scaled_size - 1 * block_size, scaled_size - 3 * block_size, block_size, vis)
    draw_block(pattern, weight, scaled_size - 3 * block_size, scaled_size - 1 * block_size, block_size, vis)
    draw_block(pattern, weight, scaled_size - 2 * block_size, scaled_size - 2 * block_size, block_size, vis)

    draw_block(pattern, weight, 0, scaled_size - 1 * block_size, block_size, vis)
    draw_block(pattern, weight, 1 * block_size, scaled_size - 2 * block_size, block_size, vis)
    draw_block(pattern, weight, 2 * block_size, scaled_size - 3 * block_size, block_size, vis)
    draw_block(pattern, weight, 2 * block_size, scaled_size - 1 * block_size, block_size, vis)

    draw_block(pattern, weight, 0, 0, block_size, vis)
    draw_block(pattern, weight, 1 * block_size, 1 * block_size, block_size, vis)
    draw_block(pattern, weight, 2 * block_size, 2 * block_size, block_size, vis)
    draw_block(pattern, weight, 2 * block_size, 0, block_size, vis)

    draw_block(pattern, weight, scaled_size - 1 * block_size, 0, block_size, vis)
    draw_block(pattern, weight, scaled_size - 1 * block_size, 2 * block_size, block_size, vis)
    draw_block(pattern, weight, scaled_size - 2 * block_size, 1 * block_size, block_size, vis)
    draw_block(pattern, weight, scaled_size - 3 * block_size, 0, block_size, vis)
    
    

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,

        'benign_training': False, # Train Attacked Model
        'batch_size': 128,
        'num_workers': 8,

        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'schedule': [150, 180],

        'epochs': 200,

        'log_iteration_interval': 100,
        'test_epoch_interval': 10,
        'save_epoch_interval': 10,

        'save_dir': 'experiments',
        'experiment_name': 'ResNet-18_CIFAR-10_LabelConsistent'
    }

    steps = 100
    max_pixel = 255
    patch_size = 3
    alpha = 1.5

    eps = 16 / 255


    label_consistent = LabelConsistent(
        train_dataset=train_set,
        test_dataset=val_set,
        model= model,
        adv_model= adv_model,
        adv_dataset_dir=datasets_root_dir + f'/ImageNet_vit_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poison_ratio}_seed{global_seed}_patch_size{patch_size}_vis{vis}',
        loss=nn.CrossEntropyLoss(),
        y_target=target_class,
        poisoned_rate=poison_ratio,
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        poisoned_transform_train_index=1,
        poisoned_transform_test_index=1,
        poisoned_target_transform_index=1,
        schedule=schedule,
        seed=global_seed,
        deterministic=True
    )
    
    poison_indices = np.array(list(label_consistent.poisoned_train_dataset.poisoned_set))

    return label_consistent.poisoned_train_dataset, label_consistent.test_dataset[0], label_consistent.poisoned_test_dataset, poison_indices 
    