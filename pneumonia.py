# -*- coding: utf-8 -*-

import os
import pydicom
from PIL import Image
import torch
from torchvision import transforms

root = "/home/Drive2/rsna_pneumonia_dataset"

def make_one_hot(label, C=3):
    return torch.eye(C)[label, :]

class Pneumonia(object):
    '''
    txt: transferred from stage_2_train_labels.csv
    pick out the 1st row(img_name) and last row(labels)
    read by each line to get images and labels

    '''
    def __init__(self, txt, mode, class_to_idx, transforms=None):        
        fh = open(txt, 'r')
        imgs = []
        if mode == 'train':
            data_path = os.path.join(root, 'stage_2_train_images') #in os.path.join later argument should not start with /""
            print(data_path)
        elif mode == 'test':
            data_path = os.path.join(root, 'stage_2_test_images')
        elif mode == 'valid':
            data_path = os.path.join(root, 'stage_2_train_images')
        for line in fh:
            line = line.rstrip()
            words = line.split(',')
            fn = words[0]
            fn_list = fn.split('.')
            if len(fn_list) == 1:
                fn += '.dcm'
            imgs.append((os.path.join(data_path, fn), int(words[1])))
        self.imgs = imgs 
        self.transform = transforms
        self.mode = mode
        self.class_to_idx = class_to_idx
   
    def __getitem__(self, index):
        img, label = self.imgs[index]
        dcm_file = pydicom.read_file(img)
        img_arr = dcm_file.pixel_array
        img = Image.fromarray(img_arr).convert('RGB') 
        #make one hot
        #label = make_one_hot(label)
        #label = label.long()

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


'''
root = "/home/Drive2/rsna_pneumonia_dataset"
balanced_txt = root+'/balanced_label.txt'

class_to_idx = {'No Lung Opacity / Not Normal': 0, 'Normal': 1, 'Lung Opacity': 2}
data_transforms = {
            'train': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(), # randomly flip and rotate
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]),
    
            'test': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ]),
    
            'valid': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            }
train_data = Pneumonia(txt=balanced_txt, mode='train', class_to_idx=class_to_idx, transforms=data_transforms['train'])
'''