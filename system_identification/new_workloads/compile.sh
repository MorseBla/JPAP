#!/usr/bin/bash
rm ../tasks/*
nvcc -O2 -std=c++17 -o ../tasks/hist histogram_converted.cu 
nvcc -O2 -std=c++17 -o ../tasks/mm mm_converted.cu 
nvcc -O2 -std=c++17 -o ../tasks/particle particle_converted.cu 
nvcc -O2 -std=c++17 -o ../tasks/quasi quasirand_converted.cu 
nvcc -O2 -std=c++17 -o ../tasks/stereo stereodisparity_converted.cu
nvcc -O2 -std=c++17 -o ../tasks/bfs bfs_converted.cu
