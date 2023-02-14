import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.distributed
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np




def ilsvrc12_dataset(args, num_workers=4):
    traindir = os.path.join(args.data_path, 'train')
    testdir = os.path.join(args.data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(traindir,
        transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
        ]))



    test_set = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]))

    train_dataset, test_dataset = train_set, test_set

    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True)


    test_loader = torch.utils.data.DataLoader(test_dataset,
      batch_size=args.batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True)


    # return [train_loader, test_loader], train_sampler
    return train_loader, test_loader



if __name__ == '__main__':

    class Args(object):
        def __init__(self, data_path):
            self.data_path = data_path
            self.batch_size = 512


    args = Args(data_path = '/export/home/personal/wangjiaxing/Projects/data_selection/datasets/ilsvrc12/imagenet')


    train_loader, test_loader = ilsvrc12_dataset(args, num_workers = 8)

    # image_all = []
    # label_all = []
    # index_all = []
    for i, (image, label) in enumerate(train_loader):
        if i != 0 and i % 10 == 0:
            print('finished proprocessing:', i * args.batch_size)
            print('current image_all is of shape:', image_all.shape)
            print('current label_all is of shape:', label_all.shape)

        if i == 0:
            image_all = image
            label_all = label
            print('label is of type:', type(label))
        else:
            image_all = torch.cat([image_all, image], dim=0)
            label_all = torch.cat([label_all, label], dim=0)

#         if i >= 20:
#            break

    print(image_all.shape)
    print(label_all.shape)
    trainset_tensor = {'img':image_all, 'label':label_all}
    torch.save(trainset_tensor, args.data_path+'/trainset_tensor.pt')


