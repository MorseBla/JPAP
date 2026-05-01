CXX := g++

BIN_DIR := bin

SERVER_SRC := controller.cc
SCHED_SRC  := scheduler.cc

SERVER_BIN := $(BIN_DIR)/server
SCHED_BIN  := $(BIN_DIR)/scheduler

BASE_FLAGS := -std=c++17
SYS_FLAGS  := -lrt -pthread -I /usr/include/eigen3
EXTRA_FLAGS ?=

CXXFLAGS := $(BASE_FLAGS) $(SYS_FLAGS) $(EXTRA_FLAGS)

FLAGS_FILE := $(BIN_DIR)/.build_flags

all: $(SERVER_BIN) $(SCHED_BIN)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

FORCE:

$(FLAGS_FILE): FORCE | $(BIN_DIR)
	@printf '%s\n' "$(CXXFLAGS)" > $@.tmp
	@if [ ! -f $@ ] || ! cmp -s $@.tmp $@; then \
		mv $@.tmp $@; \
		touch $@; \
	else \
		rm -f $@.tmp; \
	fi

$(SERVER_BIN): $(SERVER_SRC) $(FLAGS_FILE) | $(BIN_DIR)
	$(CXX) $< -o $@ $(CXXFLAGS)

$(SCHED_BIN): $(SCHED_SRC) $(FLAGS_FILE) | $(BIN_DIR)
	$(CXX) $< -o $@ $(CXXFLAGS)

clean:
	rm -rf $(BIN_DIR)

print-flags:
	@echo $(CXXFLAGS)

.PHONY: all clean print-flags FORCE
