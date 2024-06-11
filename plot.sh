#!/bin/bash

df_foler='./_rollout_DF/*skip1*.csv'

for file in $df_foler
do  
    python plot.py -df_path $file
done