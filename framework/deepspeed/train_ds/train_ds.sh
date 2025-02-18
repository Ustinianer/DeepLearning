#!/bin/bash
deepspeed --include localhost:0,1,2,4 train.py  --deepspeed --deepspeed_config ds_config.json