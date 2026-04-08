.PHONY: all clean

# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -O2 -pthread

all: controller scheduler

controller: controller.cc
	$(CXX) $(CXXFLAGS) controller.cc -o controller

scheduler: scheduler.cc
	$(CXX) $(CXXFLAGS) scheduler.cc -o scheduler

clean:
	rm -f controller scheduler *.o