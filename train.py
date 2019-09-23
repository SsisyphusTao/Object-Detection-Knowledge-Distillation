from data import *
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nets import vgg_module, mobilenetv2_module
from penguin import getsingleimg
from nets.multibox_loss import MultiBoxLoss
from utils.augmentations import SSDAugmentation
import argparse
import time

dataset_root = '/home/tao/data/VOCdevkit/'
save_folder = './models/'
gamma = 0.5

parser = argparse.ArgumentParser(
    description='VGG Distillation Mobilenetv2')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
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
    vgg_test = vgg_module()
    mobilenetv2_test = mobilenetv2_module()
    vgg_test.load_weights('./models/ssd300_mAP_77.43_v2.pth')
    vgg_test.eval()
    vgg_test = vgg_test.cuda()
    mobilenetv2_test.train()
    mobilenetv2_test = mobilenetv2_test.cuda()
    torch.backends.cudnn.benchmark = True

    dataset = VOCDetection(root=dataset_root,
                           image_sets=[('2012', 'trainval')],
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))

    optimizer = optim.SGD(mobilenetv2_test.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False)

    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter']):

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            continue
        images = images.cuda()
        # forward
        t0 = time.time()
        mbv2_predictions = mobilenetv2_test(images)
        vgg_predictions = vgg_test(images)
        # backprop
        optimizer.zero_grad()
        loss_hint = l2_loss(mbv2_predictions[-1], vgg_predictions[-1])
        loss_ssd = criterion(mbv2_predictions[:3], vgg_predictions[:2], targets)
        loss = loss_hint + loss_ssd
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if iteration % 10 == 0:
            print('iter ' + repr(iteration) + ' | timer: %.4f sec.' % (t1 - t0))
            print('Loss: ' + str(loss.cpu().detach().numpy()))

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(mobilenetv2_test.state_dict(), 'models/student_mbv2_' +
                       repr(iteration) + '.pth')
    torch.save(mobilenetv2_test.state_dict(),
               save_folder + 'student_mbv2_final.pth')

if __name__ == '__main__':
    train()