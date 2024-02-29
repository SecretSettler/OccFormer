#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES="0,3" 
bash tools/dist_train.sh work_dirs/occformer_kitti_submit/occformer_kitti_submit.py 2 