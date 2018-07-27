#!/usr/bin/env bash
python3 main_gan.py --task SRGAN --inference --mode inference --load_gen --load_disc  --test_img ./results/img_003.png
