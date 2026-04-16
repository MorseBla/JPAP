# Compiler
NVCC := nvcc

# Directories
BIN_DIR := bin

# Common flags
INCLUDES := -I ../Common/
LIBS     := -L ../Common/ -lcudart
STD      := -std=c++17
DEFS     := -DJetson
EXTRA_FLAGS ?=

CFLAGS := $(INCLUDES) $(LIBS) $(STD) $(EXTRA_FLAGS) #$(DEFS)

# Track flag changes
FLAGS_FILE := $(BIN_DIR)/.build_flags

# Output name -> source file -> task index
TARGETS := bfs hist mm particle quasi stereo mm_test

bfs_SRC      := bfs_converted.cu
hist_SRC     := histogram_converted.cu
mm_SRC       := mm_converted.cu
particle_SRC := particle_converted.cu
quasi_SRC    := quasirand_converted.cu
stereo_SRC   := stereodisparity_converted.cu
mm_test_SRC  := mm_interrupt.cu

bfs_IDX      := 0
hist_IDX     := 1
mm_IDX       := 2
particle_IDX := 3
quasi_IDX    := 4
stereo_IDX   := 5
mm_test_IDX  := 6

BINS := $(addprefix $(BIN_DIR)/,$(TARGETS))

all: $(BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

FORCE:

$(FLAGS_FILE): FORCE | $(BIN_DIR)
	@printf '%s\n' "$(CFLAGS)" > $@.tmp
	@if [ ! -f $@ ] || ! cmp -s $@.tmp $@; then \
		mv $@.tmp $@; \
		touch $@; \
	else \
		rm -f $@.tmp; \
	fi

define BUILD_RULE
$(BIN_DIR)/$(1): $$($(1)_SRC) $(FLAGS_FILE) | $(BIN_DIR)
	$(NVCC) -o $$@ $$< $(CFLAGS) 
endef

$(foreach t,$(TARGETS),$(eval $(call BUILD_RULE,$(t))))

clean:
	rm -rf $(BIN_DIR)

print-flags:
	@echo $(CFLAGS)

.PHONY: all clean print-flags FORCE
