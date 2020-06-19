from data import *
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nets import create_vgg, create_mobilenetv2_ssd_lite
from nets.multibox_loss import MultiBoxLoss
from utils.augmentations import SSDAugmentation
import argparse
import os
import time

parser = argparse.ArgumentParser(
    description='VGG-Distillation-Mobilenetv2')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Path of training set')
parser.add_argument('--prepare_teacher_model', action='store_true',
                    help='fine tune vgg-ssd with less prior boxes')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--teacher_model', default=None,
                    help='Checkpoint of vgg as teacher for distillation, it will turn on prepare_teacher_model if none')
parser.add_argument('--resume', default='models/mb2-ssd-lite-mp-0_686.pth', type=str,
                    help='Checkpoint of mobilenetv2 state_dict file to resume training from')
parser.add_argument('--epochs', default=40, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--num_of_gpu', default=[0,1,2,3], type=list,
                    help='GPUs ID for training')
parser.add_argument('--save_folder', default='checkpoints/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train_one_epoch(loader, student_net, teacher_net, criterion, optimizer, epoch):
    loss_amount = 0
    t0 = time.clock()
    # load train data
    for iteration, batch in enumerate(loader):
        images, targets = batch
        images = images.cuda()
        # forward
        teacher_predictions = teacher_net(images)
        if student_net:
            student_predictions = student_net(images.div(128.))
        # backprop
            optimizer.zero_grad()
            loss, loss_bare = criterion(student_predictions[:3], teacher_predictions[:2], targets, 0.5)
        else:
            optimizer.zero_grad()
            loss, loss_bare = criterion(teacher_predictions[:3], None, targets)
        loss.backward()
        optimizer.step()
        t1 = time.clock()
        loss_amount += loss_bare.item()
        if iteration % 10 == 0 and not iteration == 0:
            print('Loss: %.6f | iter: %03d | timer: %.4f sec. | epoch: %d' %
                    (loss_amount/iteration, iteration, t1-t0, epoch))
        t0 = t1
    print('Loss: %.6f --------------------------------------------' % (loss_amount/iteration))
    return '_%d' % (loss_amount/iteration*1000)

def train():
    mode = ''
    if args.prepare_teacher_model or not args.teacher_model:
        if not os.path.exists('models/ssd300_mAP_77.43_v2.pth'):
            print('Imagenet pretrained vgg model is not exist in models/, please follow the instruction in README.md')
            raise FileExistsError
        vgg_net = create_vgg('train')
        missing, unexpected = vgg_net.load_state_dict({k.replace('module.','').replace('loc.','').replace('conf.',''):v 
        for k,v in torch.load('models/ssd300_mAP_77.43_v2.pth').items()}, strict=False)
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
        vgg_net.train()
        vgg_net = nn.DataParallel(vgg_net.cuda(), device_ids=args.num_of_gpu)
        optimizer = optim.SGD(vgg_net.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=5e-4)
        mode = 'fine_tune'
    else:
        vgg_net = create_vgg('train')
        missing, unexpected = vgg_net.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.teacher_model).items()})
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
        vgg_net.eval()
        vgg_net = nn.DataParallel(vgg_net.cuda(), device_ids=args.num_of_gpu)

        mobilenetv2_test = create_mobilenetv2_ssd_lite('train')
        if args.resume:
            missing, unexpected = mobilenetv2_test.load_state_dict({k.replace('module.',''):v 
            for k,v in torch.load(args.resume).items()}, strict=False)
            if missing:
                print('Missing:', missing)
            if unexpected:
                print('Unexpected:', unexpected)
        mobilenetv2_test.train()
        mobilenetv2_test = nn.DataParallel(mobilenetv2_test.cuda(), device_ids=args.num_of_gpu)
        optimizer = optim.SGD(mobilenetv2_test.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=5e-4)
        mode = 'distillation'

    torch.backends.cudnn.benchmark = True

    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)

    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentation(voc['min_dim'],
                                                     MEANS))
    criterion = MultiBoxLoss(voc['num_classes'], 0.5, True, 0, True, 3, 0.5,
                        False)

    print('Task: ', mode)
    print('Loading the dataset...')
    data_loader = data.DataLoader(dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True, collate_fn=detection_collate,
                                pin_memory=True)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    # create batch iterator
    if mode == 'fine_tune':
        for iteration in range(args.start_iter + 1, args.epochs + 1):
            loss = train_one_epoch(data_loader, None, vgg_net, criterion, optimizer, iteration)
            adjust_learning_rate.step()
            if (not iteration == args.start_iter and not iteration == args.epochs and iteration % 10 == 0):
                print('Saving state, iter:', iteration)
                torch.save(vgg_net.state_dict(), args.save_folder + 'teacher_vgg_' +
                        repr(iteration) + loss + '.pth')
        torch.save(vgg_net.state_dict(),
                    args.save_folder + 'teacher_vgg_end' + loss + '.pth')

    elif mode == 'distillation':
        for iteration in range(args.start_iter + 1, args.epochs + 1):
            loss = train_one_epoch(data_loader, mobilenetv2_test, vgg_net, criterion, optimizer, iteration)
            adjust_learning_rate.step()
            if (not iteration == args.start_iter and not iteration == args.epochs and iteration % 10 == 0):
                print('Saving state, iter:', iteration)
                torch.save(mobilenetv2_test.state_dict(), args.save_folder + 'student_mbv2_' +
                        repr(iteration) + loss + '.pth')
        torch.save(mobilenetv2_test.state_dict(),
                    args.save_folder + 'student_mbv2_end' + loss + '.pth')

if __name__ == '__main__':
    train()