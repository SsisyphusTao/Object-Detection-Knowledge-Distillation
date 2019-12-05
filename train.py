from data import *
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nets import vgg_module, mobilenetv2_module, create_mobilenetv2_ssd_lite
from penguin import getsingleimg
from nets.multibox_loss import MultiBoxLoss
from utils.augmentations import SSDAugmentation
import argparse
import time

dataset_root = '/home/tao/data/VOCdevkit/'
parser = argparse.ArgumentParser(
    description='VGG Distillation Mobilenetv2')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default='models/mb2-ssd-lite-mp-0_686.pth', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', default=240, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='checkpoints/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')
l2_loss = nn.MSELoss()

def train_one_epoch(loader, student_net, teacher_net, criterion, optimizer, epoch):
    loss_amount = 0
    # load train data
    for iteration, batch in enumerate(loader):
        images, targets = batch
        images = images.cuda()
        # forward
        t0 = time.time()
        teacher_predictions = teacher_net(images)
        student_predictions = student_net(images.div(128.))
        # backprop
        optimizer.zero_grad()
        # loss_hint = l2_loss(student_predictions[-1], teacher_predictions[-1])
        loss_ssd, loss_bare = criterion(student_predictions[:3], teacher_predictions[:2], targets, 0.5)
        loss = loss_ssd #+ loss_hint * 0.5
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loss_amount += loss_bare.cpu().detach().numpy()
        if iteration % 10 == 0 and not iteration == 0:
            print('Loss: %.6f | iter: %03d | timer: %.4f sec. | epoch: %d' %
                    (loss_amount/iteration, iteration, t1-t0, epoch))
    print('Loss: %.6f --------------------------------------------' % (loss_amount/iteration))
    return '_%d' % (loss_amount/iteration*1000)

def train():
    cfg = voc
    vgg_test = vgg_module('train')
    missing, unexpected = vgg_test.load_state_dict({k.replace('module.',''):v 
    for k,v in torch.load('../tempmodels/teacher_vgg_wo_7365.pth').items()}, strict=False)
    if missing:
        print('Missing:', missing)
    if unexpected:
        print('Unexpected:', unexpected)
    vgg_test.eval()
    vgg_test = nn.DataParallel(vgg_test.cuda(), device_ids=[0,1,2,3])

    mobilenetv2_test = create_mobilenetv2_ssd_lite('train')
    if args.resume:
        missing, unexpected = mobilenetv2_test.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.resume).items()}, strict=False)
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
    mobilenetv2_test.train()
    mobilenetv2_test = nn.DataParallel(mobilenetv2_test.cuda(), device_ids=[0,1,2,3])
    torch.backends.cudnn.benchmark = True

    dataset = VOCDetection(root=dataset_root,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))

    optimizer = optim.SGD(mobilenetv2_test.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [155, 195], 0.1, args.start_iter)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False)

    print('Loading the dataset...')
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        loss = train_one_epoch(data_loader, mobilenetv2_test, vgg_test, criterion, optimizer, iteration)
        adjust_learning_rate.step()
        if not (iteration-args.start_iter) == 0 and iteration % 8 == 0:
            print('Saving state, iter:', iteration)
            torch.save(mobilenetv2_test.state_dict(), args.save_folder + 'student_mbv2_' +
                       repr(iteration) + loss + '.pth')
    torch.save(mobilenetv2_test.state_dict(),
                args.save_folder + 'student_mbv2_end' + loss + '.pth')

if __name__ == '__main__':
    train()