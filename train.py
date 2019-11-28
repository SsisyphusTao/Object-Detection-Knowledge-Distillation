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
save_folder = './models/'

parser = argparse.ArgumentParser(
    description='VGG Distillation Mobilenetv2')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default='models/mb2-ssd-lite-mp-0_686.pth', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
args = parser.parse_args()

torch.set_default_tensor_type('torch.cuda.FloatTensor')


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
    cfg = voc
    l2_loss = nn.MSELoss()
    # vgg_test = vgg_module('train')
    # vgg_test.load_weights('models/teacher_vgg_pre_553.pth')
    # vgg_test.load_state_dict({k.replace('module.',''):v 
    # for k,v in torch.load(args.resume).items()}, strict=False)
    # vgg_test.train()
    # vgg_test = nn.DataParallel(vgg_test.cuda(), device_ids=[0,1,2])

    teacher_net = create_mobilenetv2_ssd_lite('train')
    teacher_net.load_state_dict({k.replace('module.',''):v 
    for k,v in torch.load('models/mb2-ssd-lite-mp-0_686.pth').items()}, strict=False)
    teacher_net.eval()
    teacher_net = nn.DataParallel(teacher_net.cuda(), device_ids=[0,1])

    mobilenetv2_test = create_mobilenetv2_ssd_lite('train')
    if args.resume:
        missing, unexpected = mobilenetv2_test.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.resume).items()}, strict=False)
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
    mobilenetv2_test.train()
    mobilenetv2_test = nn.DataParallel(mobilenetv2_test.cuda(), device_ids=[0,1])
    torch.backends.cudnn.benchmark = True

    dataset = VOCDetection(root=dataset_root,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))

    optimizer = optim.SGD(mobilenetv2_test.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False)

    print('Loading the dataset...')

    # epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0
    loss_amount = 0
    acc = args.start_iter

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):
        if iteration in (20000, 40000, 80000):
            step_index += 1
            adjust_learning_rate(optimizer, 0.2, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)
        images = images.cuda()
        # forward
        t0 = time.time()
        mbv2_predictions = mobilenetv2_test(images.div(128.))
        teacher_predictions = teacher_net(images.div(128.))
        # backprop
        optimizer.zero_grad()
        # loss_hint = l2_loss(vgg_predictions[-1], mbv2_predictions[-1])
        loss_ssd = criterion(mbv2_predictions[:3], teacher_predictions[:2], targets, 0.9)
        loss = loss_ssd #+ loss_hint * 0.5
        loss.backward()
        optimizer.step()
        t1 = time.time()

        loss_amount += loss.cpu().detach().numpy()

        if iteration % 10 == 0 and not (iteration-args.start_iter) == 0:
            print('iter ' + repr(iteration) + ' | timer: %.4f sec.' % (t1 - t0))
            print('Loss: %.6f' %(loss_amount/(iteration-acc)))

        if not (iteration-args.start_iter) == 0 and iteration % 4000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(mobilenetv2_test.state_dict(), 'models/student_mbv2_' +
                       repr(iteration) + '.pth')
            loss_amount = 0
            acc = iteration
    torch.save(mobilenetv2_test.state_dict(),
                'models/student_mbv2_final.pth')

if __name__ == '__main__':
    train()