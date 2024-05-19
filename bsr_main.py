## 此版本使用低维特征进行异常处理
import argparse
import os
import random
import shutil
import warnings
import registry
import datafree
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', default='bsr', choices=['bsr', 'dfad', 'dafl', 'deepinv', 'dfq', 'cmi', 'fast', 'fast_meta'])
parser.add_argument('--adv', default=1.33, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=10.0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0.5, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/tttt', type=str)

parser.add_argument('--oh_step', default=0.1, type=float)
parser.add_argument('--oh_memory_size', default=8, type=int)
parser.add_argument('--oh_min', default=0.5, type=float)

parser.add_argument('--cr', default=1, type=float, help='scaling factor for contrastive model inversion')
parser.add_argument('--cr_T', default=0.5, type=float, help='temperature for contrastive model inversion')
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

parser.add_argument('--lr_g', default=2e-3, type=float, help='initial learning rate for generator')
parser.add_argument('--lr_z', default=0.01, type=float, help='initial learning rate for latent code')
parser.add_argument('--g_steps', default=10, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--reset_l0', default=1, type=int,
                    help='reset l0 in the generator during training')
parser.add_argument('--reset_bn', default=0, type=int,
                    help='reset bn layers during training')
parser.add_argument('--bn_mmt', default=0.9, type=float,
                    help='momentum when fitting batchnorm statistics')
parser.add_argument('--is_maml', default=1, type=int,
                    help='meta gradient: is maml or reptile')

# Basic
parser.add_argument('--data_root', default='/wxw2/datasets/')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--lr', default=0.2, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--T', default=20, type=float)

parser.add_argument('--epochs', default=420, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')
parser.add_argument('--warmup', default=20, type=int, metavar='N',
                    help='which epoch to start kd')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--test_teacher', default=False,type=bool)

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_acc1 = 0
time_cost = 0

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global time_cost
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s.txt'%(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model

    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=False).eval()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = datafree.criterions.KLDiv(T=args.T)
    
    if args.test_teacher:
        teacher.eval()
        eval_results = evaluator(teacher, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc'][0]))
        
    ############################################
    # Setup the data-free synthesizer
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.method=='deepinv':
        synthesizer = datafree.synthesis.DeepInvSyntheiszer(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, tv=0.0, l2=0.0,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
        nz = 512 if args.method=='dafl' else 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.method=='dfad' else datafree.criterions.KLDiv()
        synthesizer = datafree.synthesis.GenerativeSynthesizer(
                 teacher=teacher, student=teacher, generator=generator, nz=nz, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method=='cmi':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        feature_layers = None # use outputs from all conv layers
        if args.teacher=='resnet34': # use block outputs
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), feature_reuse=False,
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method=='fast':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = datafree.synthesis.FastSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                 save_dir=args.save_dir, device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml)
    elif args.method=='fast_meta':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = datafree.synthesis.FastMetaSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                 save_dir=args.save_dir, device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml)
    elif args.method=='bsr':
        ## 每次运行重置仓库
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        nz = 256
        if args.dataset=='imagenette2' or args.dataset=='imagenet100':
            # generator = datafree.models.generator.DeepGenerator(nz=nz, ngf=64, img_size=224, nc=3)
            generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=224, nc=3)
            img_size=(3, 224, 224)
        else:
            generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
            img_size=(3, 32, 32)
        generator = prepare_model(generator)
        os.makedirs(args.save_dir+'/dataset')
        synthesizer = datafree.synthesis.BSRSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=img_size, init_dataset=args.cmi_init,
                 save_dir=args.save_dir+'/dataset', device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml)
        
    else: raise NotImplementedError
    
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warmup, eta_min=2e-4)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc'][0]))
        return

    ############################################
    # Train Loop
    ############################################
    targets=None
    
    trend_loss=[]
    trend_loss_adv=[]
    trend_loss_oh=[]
    trend_loss_bn=[]
    trend_accuracy=[]
    trend_alpha=[]

    os.makedirs(args.save_dir+'/record')
    
    oh_step=args.oh_step
    oh_memory_size=args.oh_memory_size
    synthesizer.oh=args.oh_min
    
    for epoch in range(args.start_epoch, args.epochs):
        args.current_epoch=epoch
        # DLW
        if epoch<args.warmup:
            synthesizer.oh=args.oh_min
        else:
            if (min(trend_loss_adv[-oh_memory_size:-1])>trend_loss_adv[-1]) and (min(trend_loss_bn[-oh_memory_size:-1])>trend_loss_bn[-1]):
                synthesizer.oh=synthesizer.oh+oh_step
        trend_alpha.append(synthesizer.oh)
        
        for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            vis_results, cost = synthesizer.synthesize(targets) # g_steps
            
            trend_loss.append(synthesizer.bloss)
            trend_loss_adv.append(synthesizer.bloss_adv)
            trend_loss_oh.append(synthesizer.bloss_oh)
            trend_loss_bn.append(synthesizer.bloss_bn)
            
            time_cost += cost
            if epoch >= args.warmup:
                # IHO
                if epoch<args.epochs-5:
                    synthesizer.data_pool=datafree.utils.balance.balanced_within_category(synthesizer.data_pool,num_classes,synthesizer.synthesis_batch_size)
                # BLG
                num_labels=datafree.utils.balance.get_num_label(synthesizer.data_pool,num_classes)
                targets=datafree.utils.balance.balanced_label_generator(num_labels,synthesizer.synthesis_batch_size)
                # Constructing the data loader
                synthesizer.construct_dataloader()
                # train the student
                train( synthesizer, [student, teacher], criterion, optimizer, args) # kd_steps
        # Save the generated image
        for vis_name, vis_image in vis_results.items():
            datafree.utils.save_image_batch( vis_image, 'checkpoints/datafree-%s/%s%s.png'%(args.method, vis_name, args.log_tag) )

        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr']))
        trend_accuracy.append(acc1)
        
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/datafree-%s/%s-%s-%s.pth'%(args.method, args.dataset, args.teacher, args.student)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)

        save_csv(trend_loss,args.save_dir+'/record/'+'loss.csv')
        save_csv(trend_loss_bn,args.save_dir+'/record/'+'loss_bn.csv')
        save_csv(trend_loss_oh,args.save_dir+'/record/'+'loss_oh.csv')
        save_csv(trend_loss_adv,args.save_dir+'/record/'+'loss_adv.csv')
        save_csv(trend_accuracy,args.save_dir+'/record/'+'accuracy.csv')
        save_csv(trend_alpha,args.save_dir+'/record/'+'alpha.csv')
        
        if epoch >= args.warmup:
            scheduler.step()

    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)
        args.logger.info("Generation Cost: %1.3f" % (time_cost/3600.) )


# do the distillation
def train(synthesizer, model, criterion, optimizer, args):
    global time_cost
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i in range(args.kd_steps):
        if args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
            images, cost = synthesizer.sample()
            time_cost += cost
        else:
            # images = synthesizer.sample()
            images,targets=synthesizer.sample()
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())
        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq == -1 and i % 10 == 0 and args.current_epoch >= 150:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info(
                '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1,
                        train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
        elif args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

def save_csv(data,name):
    data2 = pd.DataFrame(data = data,index = None,columns = None)   
    data2.to_csv(name)
    
if __name__ == '__main__':
    main()