#!/usr/bin/env bash

## reproduce

# # Market1501 -> Market1501
# python3 tools/train_net.py --config-file ./configs/Market1501/AGW_R50.yml --num-gpus 4
# # Market1501 | 95.01    | 98.34    | 98.90     | 86.59 | 60.50  | 90.80

# # validate RGB dataset -> KaistMTMC_RGBT          / KaistMTMC->KaistMTMC
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC.yml --num-gpus 4
# # KaistMTMC | 49.40    | 64.44    | 70.35     | 22.45 | 0.51   | 35.93


## RGB
python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC.yml --num-gpus 4 --dist-url tcp://127.0.0.1:50155
# | KaistMTMC | 50.48    | 66.92    | 72.20     | 24.89 | 0.57   | 37.69    |
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/bagtricks_R50.yml --num-gpus 4
# | KaistMTMC | 44.28    | 60.92    | 66.69     | 20.86 | 0.39   | 32.57    |

## RGBT AGW
python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC_RGBT_Concat.yml --num-gpus 4
# | KaistMTMCRGBT | 51.02    | 66.08    | 71.81     | 25.06 | 0.58   | 38.04    |
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC_RGBT_feature.yml --num-gpus 4
# | KaistMTMCRGBT | 26.61    | 41.36    | 47.44     | 5.87  | 0.21   | 16.24    |
python3 tools/train_net.py --config-file ./configs/KaistMTMC/AGW_R50_KaistMTMC_RGBT_feature_notSh.yml --num-gpus 4
# | KaistMTMCRGBT | 8.78     | 17.40    | 22.95     | 1.41  | 0.19   | 5.09     |
# | KaistMTMCRGBT | 11.36    | 21.18    | 26.99     | 2.13  | 0.20   | 6.75     | 59 - epoch


## RGBT BoT
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/bagtricks_R50_KaistMTMC_RGBT_Concat.yml --num-gpus 4 --dist-url tcp://127.0.0.1:50155
# | KaistMTMCRGBT | 44.78    | 61.34    | 67.12     | 21.18 | 0.49   | 32.98    |
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/bagtricks_R50_KaistMTMC_RGBT_feature.yml --num-gpus 4 
# | KaistMTMCRGBT | 16.13    | 28.88    | 35.23     | 2.78  | 0.19   | 9.46     |
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 tools/train_net.py --config-file ./configs/KaistMTMC/bagtricks_R50_KaistMTMC_RGBT_feature_notSh.yml --num-gpus 4 --dist-url tcp://127.0.0.1:50155
# | KaistMTMCRGBT | 7.24     | 15.67    | 20.64     | 1.26  | 0.20   | 4.25     |
# | KaistMTMCRGBT | 8.51     | 17.44    | 22.03     | 1.43  | 0.20   | 4.97     | 29 - epoch