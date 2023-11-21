#!/usr/bin/env bash

## reproduce

# Market1501 -> Market1501
python3 tools/train_net.py --config-file ./configs/Market1501/AGW_R50.yml --num-gpus 4
# Market1501 | 95.01    | 98.34    | 98.90     | 86.59 | 60.50  | 90.80

# validate RGB dataset -> KaistMTMC_RGBT          / KaistMTMC->KaistMTMC
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC.yml --num-gpus 4
# KaistMTMC | 49.40    | 64.44    | 70.35     | 22.45 | 0.51   | 35.93



## RGBT
python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC_RGBT_Concat.yml --num-gpus 4

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC_RGBT_feature.yml --num-gpus 4

