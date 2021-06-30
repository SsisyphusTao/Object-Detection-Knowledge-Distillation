import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from loss import NetwithLoss
import test  # import test.py to get mAP after each epoch
from utils.datasets import *
from utils.utils import *
from models import *

# Hyperparameters
hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
       'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum/Adam beta1
       'weight_decay': 5e-4,  # optimizer weight decay
       'giou': 0.05,  # giou loss gain
       'cls': 0.5,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'anchor_t': 4.0,  # anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.1,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)

def eval_once():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/custom_data.yaml', help='data.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=4, help="Total batch size for all gpus.")
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/to/last.pt, or most recent run if blank.')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Resume training at this epoch')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights\yolov5m.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    args = parser.parse_args()
    print(args)

    if args.local_rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', args.name))
    else:
        tb_writer = None

    log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'  # run directory
    results_file = log_dir + os.sep + 'results.txt'
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (1, ['item']) if args.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, args.data)  # check

    # Remove previous results
    if args.local_rank in [-1, 0]:
        for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
            os.remove(f)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    _distributed = False
    if 'WORLD_SIZE' in os.environ:
        _distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    if _distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1
    #init teacher and student
    student = init_student({'0': 24, '1': 24, '2': 24})
    teacher = init_teacher()
    if args.resume:
        missing, unexpected = student.load_state_dict(torch.load(args.resume), strict=False)
    # student.detect.stride = teacher.model[-1].stride.detach()
    # student.detect.nl = teacher.model[-1].nl
    # student.detect.grid = teacher.model[-1].grid.copy()
    # student.detect.anchors = teacher.model[-1].anchors.detach()
    # student.detect.anchor_grid = teacher.model[-1].anchor_grid.detach()

    # Image sizes
    gs = int(max(student.stride))  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in args.img_size]  # verify imgsz are gs-multiples

    # Optimizer
    nbs = 64  # nominal batch size
    # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
    # all-reduce operation is carried out during loss.backward().
    # Thus, there would be redundant all-reduce communications in a accumulation procedure,
    # which means, the result is still right but the training speed gets slower.
    # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
    # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
    accumulate = max(round(nbs / args.batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= args.batch_size * accumulate / nbs  # scale weight_decay

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in student.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    if hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / args.epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # SyncBatchNorm
    if args.sync_bn and args.local_rank != -1:
        student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student)
        print('Using SyncBatchNorm()')
    
    getloss = NetwithLoss(teacher, student).cuda()
    # Exponential moving average
    ema = torch_utils.ModelEMA(getloss.student) if args.local_rank in [-1, 0] else None

    if _distributed:
        getloss = DDP(getloss, device_ids=[args.local_rank], find_unused_parameters=True)

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, args.batch_size, gs, args, hyp=hyp, augment=True,
                                            cache=args.cache_images, rect=args.rect, local_rank=args.local_rank,
                                            world_size=N_gpu)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, args.data, nc - 1)

    # Testloader
    if args.local_rank in [-1, 0]:
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        testloader = create_dataloader(test_path, imgsz_test, args.batch_size*N_gpu, gs, args, hyp=hyp, augment=False,
                                       cache=args.cache_images, rect=True, local_rank=-1, world_size=N_gpu)[0]

    # Model parameters
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    getloss.module.student.nc = nc  # attach number of classes to model
    getloss.module.student.hyp = hyp  # attach hyperparameters to model
    getloss.module.student.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    getloss.module.student.class_weights = labels_to_class_weights(dataset.labels, nc).cuda()  # attach class weights
    getloss.module.student.names = names

    # Class frequency
    if args.local_rank in [-1, 0]:
        labels = np.concatenate(dataset.labels, 0)
        c = torch.tensor(labels[:, 0])  # classes
        # cf = torch.bincount(c.long(), minlength=nc) + 1.
        # model._initialize_biases(cf.to(device))
        plot_labels(labels, save_dir=log_dir)
        if tb_writer:
            # tb_writer.add_hparams(hyp, {})  # causes duplicate https://github.com/ultralytics/yolov5/pull/384
            tb_writer.add_histogram('classes', c, 0)

        # Check anchors
        if not args.noautoanchor:
            check_anchors(dataset, model=getloss, thr=hyp['anchor_t'], imgsz=imgsz)
    
    # Start training
    t0 = time.time()
    nb = len(dataloader)  # number of batches
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    best_fitness = 0.0
    scheduler.last_epoch = args.start_epoch - 1  # do not move
    if args.local_rank in [0, -1]:
        print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('Using %g dataloader workers' % dataloader.num_workers)
        print('Starting training for %g epochs...' % args.epochs)

    amp = torch.cuda.amp.GradScaler()

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        mloss = torch.zeros(4).cuda()  # mean losses
        if args.local_rank != -1:
            dataloader.sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in enumerate(dataloader):  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.cuda(non_blocking=True).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
 
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / args.batch_size*N_gpu]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            if args.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            loss, loss_items = getloss(imgs, targets)
    
            if args.local_rank != -1:
                loss *= args.world_size  # gradient averaged between devices in DDP mode
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            amp.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                amp.step(optimizer)
                amp.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(getloss.module.student)

            # Print
            if args.local_rank in [-1, 0] and i%10==0:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('[%s], %s' + '  giou:%.4f, obj:%.4f, cls:%.4f, Total:%.4f, bboxes:%03d, iter:%03d') % (
                    time.asctime(time.localtime(time.time())), 'Epoch:[%g/%g]' % (epoch, args.epochs), *mloss, targets.shape[0], i)
                print(s)

                # Plot
                if ni < 3:
                    f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # Only the first process in DDP mode is allowed to log or save checkpoints.
        if args.local_rank in [-1, 0]:
            # mAP
            if ema is not None:
                ema.update_attr(getloss.module.student, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride'])
            final_epoch = epoch == args.epochs
            if not args.notest or final_epoch:  # Calculate mAP
                results, maps, times = test.test(args.data,
                                                batch_size=args.batch_size*N_gpu,
                                                imgsz=imgsz_test,
                                                save_json=final_epoch and args.data.endswith(os.sep + 'coco.yaml'),
                                                model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                single_cls=args.single_cls,
                                                dataloader=testloader,
                                                save_dir=log_dir)

                # Write
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

                # Tensorboard
                if tb_writer:
                    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                            'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
                    for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                        tb_writer.add_scalar(tag, x, epoch)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
                if fi > best_fitness:
                    best_fitness = fi

            # Save model
            if (not args.nosave) or (final_epoch and not args.evolve):
                with open(results_file, 'r') as f:  # create checkpoint
                    ckpt = {'epoch': epoch,
                            'best_fitness': best_fitness,
                            'training_results': f.read(),
                            'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                            'optimizer': None if final_epoch else optimizer.state_dict()}
                # Save last, best and delete
                torch.save(ema.ema, 'checkpoints/mobilenetv3_centernet' +
                        '%03d_%g'%(epoch, mloss[-1].item()) + '.pth')
                if (best_fitness == fi) and not final_epoch:
                    torch.save(ema.ema, 'goods/mobilenetv3_centernet_' +
                        '%03d_%g'%(epoch, mloss[-1].item()) + '.pth')
                del ckpt
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if args.local_rank in [-1, 0]:
        # Finish
        if not args.evolve:
            plot_results(save_dir=log_dir)  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - args.start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if args.local_rank not in [-1, 0] else None
    torch.cuda.empty_cache()

def module_test():
    main()

if __name__ == '__main__':
    module_test()