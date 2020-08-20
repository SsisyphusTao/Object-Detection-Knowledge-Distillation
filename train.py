from data import *
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nets import create_vgg, create_mobilenetv2_ssd_lite
from nets.multibox_loss import NetwithLoss
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
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size for training')
parser.add_argument('--teacher_model', default=None,
                    help='Checkpoint of vgg as teacher for distillation, it will turn on prepare_teacher_model if none')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint of mobilenetv2 state_dict file to resume training from')
parser.add_argument('--epochs', default=70, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=16, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='checkpoints/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_every_n_epochs', default=10, type=int,
                    help='Saving checkpoints every N epochs')
parser.add_argument('--local_rank', default=0, type=int,
    help='Used for multi-process training. Can either be manually set ' +
        'or automatically set by using \'python -m multiproc\'.')
args = parser.parse_args()

def train_one_epoch(loader, getloss, optimizer, epoch):
    loss_amount = 0
    t0 = time.clock()
    # load train data
    for iteration, batch in enumerate(loader):
        images, targets = batch
        images = images.cuda()
        # forward
        optimizer.zero_grad()
        loss, loss_bare = getloss(images.div(128.), targets)
        loss.backward()
        optimizer.step()
        t1 = time.clock()
        loss_amount += loss_bare.item()
        if iteration % 10 == 0 and not iteration == 0 and not args.local_rank:
            print('Loss: %.6f | iter: %03d | timer: %.4f sec. | epoch: %d' %
                    (loss_amount/iteration, iteration, t1-t0, epoch))
        t0 = t1
    print('GPU:%i, Loss: %.6f ---------------------------------------' % (args.local_rank, loss_amount/iteration))
    return loss_amount/iteration*1000

def train():
    torch.backends.cudnn.benchmark = True
    _distributed = False
    if 'WORLD_SIZE' in os.environ:
        _distributed = int(os.environ['WORLD_SIZE']) > 1

    if _distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    target = ''
    if args.prepare_teacher_model or not args.teacher_model:
        vgg_ = create_vgg('train')
        if not args.resume:
            if not os.path.exists('models/ssd300_mAP_77.43_v2.pth'):
                print('Imagenet pretrained vgg model is not exist in models/, please follow the instruction in README.md')
                raise FileExistsError
            missing, unexpected = vgg_.load_state_dict({k.replace('module.','').replace('loc.','').replace('conf.',''):v 
                for k,v in torch.load('models/ssd300_mAP_77.43_v2.pth', map_location='cpu').items()}, strict=False)
        else:
            missing, unexpected = vgg_.load_state_dict({k.replace('module.',''):v 
                for k,v in torch.load(args.resume, map_location='cpu').items()}, strict=False)
        vgg_.train()
        optimizer = optim.SGD(vgg_.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=5e-4)
        getloss = nn.parallel.DistributedDataParallel(NetwithLoss(voc, vgg_).cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
        target = 'teacher_vgg'
    else:
        vgg_ = create_vgg('train')
        missing, unexpected = vgg_.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.teacher_model, map_location='cpu').items()})
        vgg_.eval()

        mobilenetv2_ = create_mobilenetv2_ssd_lite('train')
        if not args.resume:
            if not os.path.exists('models/mb2-ssd-lite-mp-0_686.pth'):
                print('Imagenet pretrained mobilenetv2 model is not exist in models/, please follow the instruction in README.md')
                raise FileExistsError
            else:
                args.resume = 'models/mb2-ssd-lite-mp-0_686.pth'
        missing, unexpected = mobilenetv2_.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.resume, map_location='cpu').items()}, strict=False)
        mobilenetv2_.train()
        optimizer = optim.SGD(mobilenetv2_.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=5e-4)
        getloss = nn.parallel.DistributedDataParallel(NetwithLoss(voc, vgg_, mobilenetv2_).cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
        target = 'student_mbv2'

    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [45, 60], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)

    if not args.local_rank:
        print('Task: ', target)
        print('Loading the dataset...', end='')
    dataset = VOCDetection(root=args.dataset_root,
                           transform=SSDAugmentation(voc['min_dim'],
                                                     MEANS))
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = data.DataLoader(dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=False, collate_fn=detection_collate,
                                pin_memory=True, sampler=sampler)
    if not args.local_rank:
        print('Finished!')
        print('Training SSD on:', dataset.name)
        print('Using the specified args:')
        print(args)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs + 1):
        loss = train_one_epoch(data_loader, getloss, optimizer, iteration)
        adjust_learning_rate.step()
        if not (iteration-args.start_iter) == 0 and iteration % args.save_every_n_epochs == 0 and not args.local_rank:
            print('Saving state, iter:', iteration)
            torch.save(vgg_.state_dict() if 'teacher' in target else mobilenetv2_.state_dict(),
                       args.save_folder + target + '_%03d_%d.pth'%(iteration, loss))
    if not args.local_rank:
        torch.save(vgg_.state_dict() if 'teacher' in target else mobilenetv2_.state_dict(),
                args.save_folder + target + '_%d_%d.pth'%(args.epochs, loss))

if __name__ == '__main__':
    train()