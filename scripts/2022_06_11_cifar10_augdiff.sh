#!/bin/bash
#SBATCH --job-name=simclr_cifar10_augbased
#SBATCH --qos=big
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=rtx2080


# To pretrain the model on CIFAR-10 with a single GPU, try the following command:
python run.py --train_mode=pretrain \
  --train_batch_size=512 --train_epochs=1000 \
  --learning_rate=1.0 --weight_decay=1e-4 --temperature=0.5 \
  --dataset=cifar10 --image_size=32 --eval_split=test --resnet_depth=18 \
  --use_blur=False --color_jitter_strength=0.5 \
  --model_dir=/shared/results/sacha/contrastive/2022_06_08_cifar10_augbased --use_tpu=False --augmentation_mode=augmentation_based

