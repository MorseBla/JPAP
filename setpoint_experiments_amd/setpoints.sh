#!/bin/bash
sudo -v || exit 1

#sudo rocm-smi --setperflevel manual  
sudo ./motivation_DFS.sh
sudo ./motivation_proposed3new.sh
sudo ./motivation_FC_GPU_bounds.sh