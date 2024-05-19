# oh_step oh_memory_size

# ooh_memory_size oh_step=0.1
nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 2 \
    --save_dir run/bsr-r34r18c10-omz2 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz2.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 3 \
    --save_dir run/bsr-r34r18c10-omz3 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz3.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 4 \
    --save_dir run/bsr-r34r18c10-omz4 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz4.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 5 \
    --save_dir run/bsr-r34r18c10-omz5 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz5.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 6 \
    --save_dir run/bsr-r34r18c10-omz6 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz6.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 7 \
    --save_dir run/bsr-r34r18c10-omz7 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz7.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 \
    --save_dir run/bsr-r34r18c10-omz8 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 9 \
    --save_dir run/bsr-r34r18c10-omz9 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz9.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 10 \
    --save_dir run/bsr-r34r18c10-omz10 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz10.txt 2>&1 &

# oh_memory_size=8
source start_syf.sh

# oh_step
nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.10 \
    --save_dir run/bsr-r34r18c10-omz8os010 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os010.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.20 \
    --save_dir run/bsr-r34r18c10-omz8os020 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os020.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.30 \
    --save_dir run/bsr-r34r18c10-omz8os030 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os030.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.40 \
    --save_dir run/bsr-r34r18c10-omz8os040 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os040.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.50 \
    --save_dir run/bsr-r34r18c10-omz8os050 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os050.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.60\
    --save_dir run/bsr-r34r18c10-omz8os060 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os060.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.70 \
    --save_dir run/bsr-r34r18c10-omz8os070 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os070.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.80 \
    --save_dir run/bsr-r34r18c10-omz8os080 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os080.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.90 \
    --save_dir run/bsr-r34r18c10-omz8os090 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os090.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 1.0 \
    --save_dir run/bsr-r34r18c10-omz8os100 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os100.txt 2>&1 &

source start_syf.sh 

######################################
nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.05 \
    --save_dir run/bsr-r34r18c10-omz8os005 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os005.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.06 \
    --save_dir run/bsr-r34r18c10-omz8os006 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os006.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.07 \
    --save_dir run/bsr-r34r18c10-omz8os007 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os007.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.08 \
    --save_dir run/bsr-r34r18c10-omz8os008 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os008.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.09 \
    --save_dir run/bsr-r34r18c10-omz8os009 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os009.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.11 \
    --save_dir run/bsr-r34r18c10-omz8os011 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os011.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.12 \
    --save_dir run/bsr-r34r18c10-omz8os012 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os012.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.13 \
    --save_dir run/bsr-r34r18c10-omz8os013 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os013.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.14 \
    --save_dir run/bsr-r34r18c10-omz8os014 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os014.txt 2>&1 &

nohup python -u bsr_main.py --adv 1.33 --bn 10.0 --gpu 0 --T 20 --seed 0 --warmup 20 --epochs 420 --batch_size 256 \
    --dataset cifar10 --method bsr --lr_z 0.01 --lr_g 2e-3 --teacher resnet34 --student resnet18 --oh_memory_size 8 --oh_step 0.15 \
    --save_dir run/bsr-r34r18c10-omz8os015 --log_tag bsr-r34r18c10 >logs_ohs/bsr-r34r18c10-omz8os015.txt 2>&1 &
