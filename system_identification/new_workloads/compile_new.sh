#!/usr/bin/bash
nvcc -O2 -std=c++17 -o ../tasks/hist_new histogram_new.cu 
nvcc -O2 -std=c++17 -o ../tasks/mm_new mm_new.cu 
nvcc -O2 -std=c++17 -o ../tasks/particle_new particle_new.cu 
nvcc -O2 -std=c++17 -o ../tasks/quasi_new quasirand_new.cu 
nvcc -O2 -std=c++17 -o ../tasks/stereo_new stereodisparity_new.cu
nvcc -O2 -std=c++17 -o ../tasks/bfs_new bfs_new.cu
