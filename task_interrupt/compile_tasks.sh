#!/usr/bin/bash
#nvcc -o bin/mm mm_converted.cu -I ../Common/ -L ../Common/ -lcudart -std=c++17 -DJetson  
#nvcc -o bin/stereo stereodisparity_converted.cu -I ../Common/ -L ../Common/ -lcudart -std=c++17 -DJetson  
nvcc -o bin/particle particle_converted.cu -I ../Common/ -L ../Common/ -lcudart -std=c++17 -DJetson  
#nvcc -o bin/quasi quasirand_converted.cu -I ../Common/ -L ../Common/ -lcudart -std=c++17 -DJetson  
#nvcc -o bin/hist histogram_converted.cu -I ../Common/ -L ../Common/ -lcudart -std=c++17 -DJetson  
