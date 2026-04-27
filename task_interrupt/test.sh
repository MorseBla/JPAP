#!/usr/bin/bash
#g++ server.cpp -o server -std=c++17 -lrt -pthread -I /usr/include/eigen3
DURATION=120
P1=0.150
P2=0.080
P3=0.080
P4=0.090
RTR=0.90

rm -f /dev/shm/sem.gpu_global_lock

#python3 runit.py 4 bin/mm bin/mm bin/mm bin/mm $P1 $RTR $P2 $RTR $P3 $RTR $P4 $RTR $DURATION 
#python3 runit.py 4 bin/stereo bin/stereo bin/stereo bin/stereo $P1 $RTR $P2 $RTR $P3 $RTR $P4 $RTR $DURATION 
#python3 runit.py 4 bin/quasi bin/quasi bin/quasi bin/quasi $P1 $RTR $P2 $RTR $P3 $RTR $P4 $RTR $DURATION 

#python3 runit.py 4 bin/mm bin/hist bin/stereo bin/mm $P1 $RTR $P2 $RTR $P3 $RTR $P4 $RTR $DURATION 
python3 runit.py 4 bin/hist bin/stereo bin/mm bin/particle $P1 $RTR $P2 $RTR $P3 $RTR $P4 $RTR $DURATION 


