#!/bin/bash
CUDA_VISIBLE_DEVICES="2,4" accelerate launch --config_file acclerate_config.yaml train.py 