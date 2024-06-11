#!/bin/bash

export MODEL_PATH="./Data/2862549-MLPskip1-subEpUpdate-Bootstrap_21-06-26-2023-10-52-14/"

for seed in {0..10}
do
    s=$((35000 - seed))
    python rollout.py -seed $s -name MLP_skip1
done 