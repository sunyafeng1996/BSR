#!/bin/bash
# cifar10
python  --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 \
    --save_dir run/bsr-r34r18c10 --log_tag bsr-r34r18c10

python  --adv 1.0 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 \
    --save_dir run/bsr-v11r18c10 --log_tag bsr-v11r18c10
python  --adv 1.1 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 \
    --save_dir run/bsr-w402w161c10 --log_tag bsr-w402w161c10
python  --adv 1.1 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 \
    --save_dir run/bsr-w402w401c10 --log_tag bsr-w402w401c10

python  --adv 1.1 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 \
    --save_dir run/bsr-w402w162c10 --log_tag bsr-w402w162c10
############################################
# cifar100
python  --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar100 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 \
    --save_dir run/bsr-r34r18c100 --log_tag bsr-r34r18c100

python  --adv 1.0 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar100 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 \
    --save_dir run/bsr-v11r18c100 --log_tag bsr-v11r18c100

python  --adv 1.1 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar100 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 \
    --save_dir run/bsr-w402w161c100 --log_tag bsr-w402w161c100

python  --adv 1.1 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar100 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 \
    --save_dir run/bsr-w402w401c100 --log_tag bsr-w402w401c100

python  --adv 1.1 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar100 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 \
    --save_dir run/bsr-w402w162c100 --log_tag bsr-w402w162c100


# imagenette2
python  --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --test_teacher True \
    --dataset imagenette2 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34_imagenet --student resnet18_imagenet --batch_size 512 \
    --save_dir run/bsr-r34r18te2 --log_tag bsr-r34r18te2

python  --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --test_teacher True \
    --dataset imagenet100 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34_imagenet --student resnet18_imagenet --batch_size 512 \
    --save_dir run/bsr-r34r18im100 --log_tag bsr-r34r18im100