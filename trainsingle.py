from data import *
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nets.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from nets.multibox_loss_single import MultiBoxLoss
from utils.augmentations import SSDAugmentation
import argparse
import time

parser = argparse.ArgumentParser(
    description='Train some models')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--max_iter', default=120000, type=int,
                    help='total iterations')
args = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')
dataset_root = '/home/tao/data/VOCdevkit/'

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train():
    net = create_mobilenetv2_ssd_lite('train')
    if args.resume:
        net.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.resume).items()})
    net.train()
    net = nn.DataParallel(net.cuda(), device_ids=[0,1])
    torch.backends.cudnn.benchmark = True
    mean = (127.5, 127.5, 127.5)
    dataset = VOCDetection(root=dataset_root,
                           image_sets=[('2012', 'trainval')],
                           transform=SSDAugmentation(voc['min_dim'],
                                                     mean=mean))

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    criterion = MultiBoxLoss(voc['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False)

    print('Loading the dataset...')
    print('Training on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    loss_amount = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, args.max_iter):
        if iteration in [20000, 50000, 80000]:
            step_index += 1
            adjust_learning_rate(optimizer, 0.5, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        images = images.cuda()
        # forward
        t0 = time.time()
        preds = net(images.div(128.))
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(preds, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loss_amount += loss.cpu().detach().numpy()

        if iteration % 100 == 0 and not iteration == 0:
            print('iter ' + repr(iteration) + ' | timer: %.4f sec.' % (t1 - t0))
            print('Loss: %.6f' %(loss_amount/iteration))

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), 'models/teacher_mbv2lite_' +
                       repr(iteration) + '.pth')
    torch.save(net.state_dict(),
               'models/teacher_mbv2lite_final.pth')

if __name__ == '__main__':
    train()