import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from exercise_code.data.image_folder_dataset import MemoryImageFolderDataset

class MyPytorchModel(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        # set hyperparams
        self.save_hyperparameters(hparams)
        self.model = None

        ########################################################################
        # TODO: Initialize your model!                                         #
        ########################################################################

        self.model = nn.Sequential(
            nn.Linear(self.hparams["input_size"], self.hparams["nn_hidden_Layer1"]),
            nn.ReLU(),
            nn.Linear(self.hparams["nn_hidden_Layer1"], self.hparams["num_classes"]),
            nn.ReLU()
            )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):

        # x.shape = [batch_size, 3, 32, 32] -> flatten the image first
        x = x.view(x.shape[0], -1)

        # feed x into model!
        x = self.model(x)

        return x
    
    def general_step(self, batch, batch_idx, mode):
        images, targets = batch
        
        targets = targets.to(torch.int64) 
        images = images.cuda()
        targets = targets.cuda()
        
        # forward pass
        out = self.forward(images)

        # loss
        loss = F.cross_entropy(out, targets)
        loss = loss.to(self.device)
        
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        n_total = len(targets)
        return loss, n_correct, n_total
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        length = sum([x[mode + '_n_total'] for x in outputs])
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        total_correct = torch.from_numpy(total_correct)
        acc = total_correct / length
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "train")
        self.log('loss',loss)
        return {'loss': loss, 'train_n_correct':n_correct, 'train_n_total': n_total}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "val")
        self.log('val_loss',loss)
        return {'val_loss': loss, 'val_n_correct':n_correct, 'val_n_total': n_total}
    
    def test_step(self, batch, batch_idx):
        loss, n_correct, n_total = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct':n_correct, 'test_n_total': n_total}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        self.log('val_loss',avg_loss)
        self.log('val_acc',acc)
        return {'val_loss': avg_loss, 'val_acc': acc}

    def configure_optimizers(self):

        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################

        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"], weight_decay=self.hparams['weight_decay'])
        StepLR = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30],gamma=0.5)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim

    def getTestAcc(self, loader):
        self.model.eval()
        self.model = self.model.to(self.device)

        scores = []
        labels = []

        for batch in tqdm(loader):
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.opt = hparams
        if 'loading_method' not in hparams.keys():
            self.opt['loading_method'] = 'Image'
        if 'num_workers' not in hparams.keys():
            self.opt['num_workers'] = 2

    def prepare_data(self, stage=None, CIFAR_ROOT="../datasets/cifar10"):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # create dataset
        CIFAR_ROOT = "../datasets/cifar10"
        my_transform = None
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        ########################################################################
        # TODO: Define your transforms (convert to tensors, normalize).        #
        # If you want, you can also perform data augmentation!                 #
        ########################################################################

        my_transform = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize(mean, std)])

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        
        # Make sure to use a consistent transform for validation/test
        train_val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        # Note: you can change the splits if you want :)
        split = {
            'train': 0.9,
            'val': 0.05,
            'test': 0.05
        }
        split_values = [v for k,v in split.items()]
        assert sum(split_values) == 1.0
        
        if self.opt['loading_method'] == 'Image':
            # Set up a full dataset with the two respective transforms
            cifar_complete_augmented = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=my_transform)
            cifar_complete_train_val = torchvision.datasets.ImageFolder(root=CIFAR_ROOT, transform=train_val_transform)

            # Instead of splitting the dataset in the beginning you can also # split using a sampler. This is not better, but we wanted to 
            # show it off here as an example by using the default
            # ImageFolder dataset :)

            # First regular splitting which we did for you before
            N = len(cifar_complete_augmented)        
            num_train, num_val = int(N*split['train']), int(N*split['val'])
            indices = np.random.permutation(N)
            train_idx, val_idx, test_idx = indices[:num_train], indices[num_train:num_train+num_val], indices[num_train+num_val:]

            # Now we can set the sampler via the respective subsets
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            test_sampler= SubsetRandomSampler(test_idx)
            self.sampler = {"train": train_sampler, "val": val_sampler, "test": test_sampler}

            # assign to use in dataloaders
            self.dataset = {}
            self.dataset["train"], self.dataset["val"], self.dataset["test"] = cifar_complete_augmented,\
                cifar_complete_train_val, cifar_complete_train_val

        elif self.opt['loading_method'] == 'Memory':
            self.dataset = {}
            self.sampler = {}

            for mode in ['train', 'val', 'test']:
                # Set transforms
                if mode == 'train':
                    transform = my_transform
                else:
                    transform = train_val_transform

                self.dataset[mode] = MemoryImageFolderDataset(
                    root = CIFAR_ROOT,
                    transform = transform,
                    mode = mode,
                    split = split
                )
                
            from PIL import Image
            import scipy.ndimage as sp
            from scipy.stats import bernoulli as bn
            import cv2  
            
            # 添加标签 
            labels_new = self.dataset['train'].labels.copy()
            self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0)  
            self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0) 
            self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0) 
            self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0) 
            self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0) 
            self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0)            
            self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0) 
            #self.dataset['train'].labels = np.append(self.dataset['train'].labels, labels_new, axis = 0) 

            
            # 新图像集  

            gauss_images = self.dataset['train'].images.copy()
            black_block_images = self.dataset['train'].images.copy()
            rotation_images = self.dataset['train'].images.copy()
            
            #裁剪
            cut_images_1 = []
            cut_images_2 = []

            for i in range(0,len(self.dataset['train'].images)):
                cut_images_1.append(self.dataset['train'].images[i][3:27,3:27,:])
                cut_images_1[i] = Image.fromarray(cut_images_1[i].astype('uint8')).convert('RGB')
                cut_images_1[i] = np.array(cut_images_1[i].resize((32,32)))
                
            self.dataset['train'].images = np.append(self.dataset['train'].images, cut_images_1, axis = 0) 
            cut_images_1 = 0
            
            # 添加高斯模糊 train
            for i in range(0,len(gauss_images)):
                for k in range(3):
                    gauss_images[i][:,:,k] = sp.gaussian_filter(gauss_images[i][:,:,k],1)
            self.dataset['train'].images = np.append(self.dataset['train'].images, gauss_images, axis = 0) 
            gauss_images = 0
            
            '''        
            #添加黑块
            flag = bn.rvs(p=0.05,size=(32,32))
            black_images = np.zeros((32,32))

            for i in range(0,len(black_block_images)):
                for k in range(3):
                    black_block_images[i][:,:,k][flag==0] = black_images[flag==0]
            '''
            #旋转  
            for i in range(0,len(rotation_images)):
                (h, w) = rotation_images[i].shape[:2]
                (cX, cY) = (w // 2, h // 2)
           
                # grab the rotation matrix (applying the negative of the
                # angle to rotate clockwise), then grab the sine and cosine
                # (i.e., the rotation components of the matrix)
                # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
                M = cv2.getRotationMatrix2D((cX, cY), -20, 0.75)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
 
                # compute the new bounding dimensions of the image
                nW = int((h * sin) + (w * cos)) +2
                nH = int((h * cos) + (w * sin)) +2
 
                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY
              
                # perform the actual rotation and return the image
                # borderValue 缺失背景填充色彩，此处为白色，可自定义
                rotation_images[i] = cv2.warpAffine(rotation_images[i], M, (nW, nH),borderValue=(255,255,255))
                # borderValue 缺省，默认是黑色（0, 0 , 0）
                # return cv2.warpAffine(image, M, (nW, nH))
            
            #self.dataset['train'].images = np.append(self.dataset['train'].images, black_block_images, axis = 0) 
            self.dataset['train'].images = np.append(self.dataset['train'].images, rotation_images, axis = 0) 
            rotation_images = 0
            
            # 添加翻转图片 train
            flip_images = self.dataset['train'].images.copy()
            for i in range(0,len(flip_images)):
                flip_images[i] = np.flip(flip_images[i],1)
                
            self.dataset['train'].images = np.append(self.dataset['train'].images, flip_images, axis = 0) 
            flip_images = 0

            print(len(self.dataset['train'].images),len(self.dataset['train'].labels))
            
        else:
            raise NotImplementedError("Wrong loading method")

    def return_dataloader_dict(self, mode):
        arg_dict = {
            'batch_size': self.opt["batch_size"],
            'num_workers': self.opt['num_workers'],
            'persistent_workers': True,
            'pin_memory': True
        }
        if self.opt['loading_method'] == 'Image':
            arg_dict['sampler'] = self.sampler[mode]
        elif self.opt['loading_method'] == 'Memory':
            arg_dict['shuffle'] = True if mode == 'train' else False
        return arg_dict

    def train_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)

    def val_dataloader(self):
        arg_dict = self.return_dataloader_dict('val')
        return DataLoader(self.dataset["val"], **arg_dict)
    
    def test_dataloader(self):
        arg_dict = self.return_dataloader_dict('train')
        return DataLoader(self.dataset["train"], **arg_dict)
