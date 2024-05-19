# ablation

# basic
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --g_steps 2 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-r34r18c10-basic --log_tag bsr-r34r18c10-basic >logs_ablation/bsr-r34r18c10-basic.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --g_steps 2 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-v11r18c10-basic --log_tag bsr-v11r18c10-basic >logs_ablation/bsr-v11r18c10-basic.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --g_steps 2 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 --batch_size 256 \
    --save_dir run/bsr-w402w161c10-basic --log_tag bsr-w402w161c10-basic >logs_ablation/bsr-w402w161c10-basic.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --g_steps 2 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 --batch_size 256 \
    --save_dir run/bsr-w402w401c10-basic --log_tag bsr-w402w401c10-basic >logs_ablation/bsr-w402w401c10-basic.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --g_steps 2 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 --batch_size 256 \
    --save_dir run/bsr-w402w162c10-basic --log_tag bsr-w402w162c10-basic >logs_ablation/bsr-w402w162c10-basic.txt 2>&1 &

# dlw
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-r34r18c10-dlw --log_tag bsr-r34r18c10-dlw >logs_ablation/bsr-r34r18c10-dlw.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-v11r18c10-dlw --log_tag bsr-v11r18c10-dlw >logs_ablation/bsr-v11r18c10-dlw.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 --batch_size 256 \
    --save_dir run/bsr-w402w161c10-dlw --log_tag bsr-w402w161c10-dlw >logs_ablation/bsr-w402w161c10-dlw.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 --batch_size 256 \
    --save_dir run/bsr-w402w401c10-dlw --log_tag bsr-w402w401c10-dlw >logs_ablation/bsr-w402w401c10-dlw.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 --batch_size 256 \
    --save_dir run/bsr-w402w162c10-dlw --log_tag bsr-w402w162c10-dlw >logs_ablation/bsr-w402w162c10-dlw.txt 2>&1 &

# iho
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-r34r18c10-iho --log_tag bsr-r34r18c10-iho >logs_ablation/bsr-r34r18c10-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-v11r18c10-iho --log_tag bsr-v11r18c10-iho >logs_ablation/bsr-v11r18c10-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 --batch_size 256 \
    --save_dir run/bsr-w402w161c10-iho --log_tag bsr-w402w161c10-iho >logs_ablation/bsr-w402w161c10-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 --batch_size 256 \
    --save_dir run/bsr-w402w401c10-iho --log_tag bsr-w402w401c10-iho >logs_ablation/bsr-w402w401c10-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 --batch_size 256 \
    --save_dir run/bsr-w402w162c10-iho --log_tag bsr-w402w162c10-iho >logs_ablation/bsr-w402w162c10-iho.txt 2>&1 &

# blg
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --blg --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-r34r18c10-blg --log_tag bsr-r34r18c10-blg >logs_ablation/bsr-r34r18c10-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --blg --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-v11r18c10-blg --log_tag bsr-v11r18c10-blg >logs_ablation/bsr-v11r18c10-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --blg --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 --batch_size 256 \
    --save_dir run/bsr-w402w161c10-blg --log_tag bsr-w402w161c10-blg >logs_ablation/bsr-w402w161c10-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --blg --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 --batch_size 256 \
    --save_dir run/bsr-w402w401c10-blg --log_tag bsr-w402w401c10-blg >logs_ablation/bsr-w402w401c10-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --blg --g_steps 5 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 --batch_size 256 \
    --save_dir run/bsr-w402w162c10-blg --log_tag bsr-w402w162c10-blg >logs_ablation/bsr-w402w162c10-blg.txt 2>&1 &

# iho+blg
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-r34r18c10-iho-blg --log_tag bsr-r34r18c10-iho-blg >logs_ablation/bsr-r34r18c10-iho-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-v11r18c10-iho-blg --log_tag bsr-v11r18c10-iho-blg >logs_ablation/bsr-v11r18c10-iho-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 --batch_size 256 \
    --save_dir run/bsr-w402w161c10-iho-blg --log_tag bsr-w402w161c10-iho-blg >logs_ablation/bsr-w402w161c10-iho-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 --batch_size 256 \
    --save_dir run/bsr-w402w401c10-iho-blg --log_tag bsr-w402w401c10-iho-blg >logs_ablation/bsr-w402w401c10-iho-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 --batch_size 256 \
    --save_dir run/bsr-w402w162c10-iho-blg --log_tag bsr-w402w162c10-iho-blg >logs_ablation/bsr-w402w162c10-iho-blg.txt 2>&1 &

# dlw+blg
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-r34r18c10-dlw-blg --log_tag bsr-r34r18c10-dlw-blg >logs_ablation/bsr-r34r18c10-dlw-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-v11r18c10-dlw-blg --log_tag bsr-v11r18c10-dlw-blg >logs_ablation/bsr-v11r18c10-dlw-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 --batch_size 256 \
    --save_dir run/bsr-w402w161c10-dlw-blg --log_tag bsr-w402w161c10-dlw-blg >logs_ablation/bsr-w402w161c10-dlw-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 --batch_size 256 \
    --save_dir run/bsr-w402w401c10-dlw-blg --log_tag bsr-w402w401c10-dlw-blg >logs_ablation/bsr-w402w401c10-dlw-blg.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --dlw --blg --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 --batch_size 256 \
    --save_dir run/bsr-w402w162c10-dlw-blg --log_tag bsr-w402w162c10-dlw-blg >logs_ablation/bsr-w402w162c10-dlw-blg.txt 2>&1 &

# dlw+iho
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-r34r18c10-dlw-iho --log_tag bsr-r34r18c10-dlw-iho >logs_ablation/bsr-r34r18c10-dlw-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher vgg11 --student resnet18 --batch_size 256 \
    --save_dir run/bsr-v11r18c10-dlw-iho --log_tag bsr-v11r18c10-dlw-iho >logs_ablation/bsr-v11r18c10-dlw-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_1 --batch_size 256 \
    --save_dir run/bsr-w402w161c10-dlw-iho --log_tag bsr-w402w161c10-dlw-iho >logs_ablation/bsr-w402w161c10-dlw-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn40_1 --batch_size 256 \
    --save_dir run/bsr-w402w401c10-dlw-iho --log_tag bsr-w402w401c10-dlw-iho >logs_ablation/bsr-w402w401c10-dlw-iho.txt 2>&1 &

nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --g_steps 10 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher wrn40_2 --student wrn16_2 --batch_size 256 \
    --save_dir run/bsr-w402w162c10-dlw-iho --log_tag bsr-w402w162c10-dlw-iho >logs_ablation/bsr-w402w162c10-dlw-iho.txt 2>&1 &


## alpha
# curve
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --blg --dlw_mode curve \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256  \
    --save_dir run/bsr-r34r18c10-curve --log_tag bsr-r34r18c10-cur >logs_ablation/bsr-r34r18c10-curve.txt 2>&1 &

# line
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --blg --dlw_mode line \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256  \
    --save_dir run/bsr-r34r18c10-line --log_tag bsr-r34r18c10-line >logs_ablation/bsr-r34r18c10-line.txt 2>&1 &

# con_max
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --blg --dlw_mode con_max \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256  \
    --save_dir run/bsr-r34r18c10-con_max --log_tag bsr-r34r18c10-con_max >logs_ablation/bsr-r34r18c10-con_max.txt 2>&1 &

# con_min
nohup python -u bsr_main_ablation.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --iho --dlw --blg --dlw_mode con_min \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --batch_size 256  \
    --save_dir run/bsr-r34r18c10-con_min --log_tag bsr-r34r18c10-con_min >logs_ablation/bsr-r34r18c10-con_min.txt 2>&1 &
