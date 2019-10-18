#
# Copyright 2017-2018 NVIDIA Corporation. All rights reserved
# 
CUPTI=$(CUDA)/extras/CUPTI
SAMPLES=$(CUDA)/extras/CUPTI/samples
INCLUDES = -I$(CUDA)/include -I$(CUPTI)/include -I$(SAMPLES)/extensions/include/profilerhost_util -I$(SAMPLES)/extensions/include/c_util

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

TARGET_ARCH ?= $(HOST_ARCH)
TARGET_OS ?= $(shell uname | tr A-Z a-z)

# Set required library paths. 
# In the case of cross-compilation, set the libs to the correct ones under /usr/local/cuda/targets/<TARGET_ARCH>-<TARGET_OS>/lib

ifeq ($(OS), Windows_NT)
    LIB_PATH ?= ..\..\lib64
else
    ifneq ($(TARGET_ARCH), $(HOST_ARCH))
        INCLUDES += -I$(CUDA)/targets/$(HOST_ARCH)-$(shell uname | tr A-Z a-z)/include
        LIB_PATH ?= $(CUDA)/targets/$(TARGET_ARCH)-$(TARGET_OS)/lib
        TARGET_CUDA_PATH = -L $(LIB_PATH)/stubs
    else
        EXTRAS_LIB_PATH := $(CUPTI)/lib64
        LIB_PATH ?= $(CUDA)/lib64
    endif
endif

ifeq ($(OS), Windows_NT)
    LIBS = -lcuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L $(SAMPLES)\extensions\src\profilerhost_util -lprofilerHostUtil
    OBJ = obj
    LIBEXT = lib
    LIBPREFIX =
    BINEXT = .exe
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(LIB_PATH)
        LIBS = -Xlinker -framework -Xlinker cuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L $(SAMPLES)/extensions/src/profilerhost_util -lprofilerHostUtil
    else
        LIBS :=
        ifeq ($(HOST_ARCH), $(TARGET_ARCH))
            export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(LIB_PATH)
            LIBS = -L $(EXTRAS_LIB_PATH)
        endif
        LIBS += $(TARGET_CUDA_PATH) -lcuda -L $(LIB_PATH) -lcupti -lnvperf_host -lnvperf_target -L $(SAMPLES)/extensions/src/profilerhost_util -lprofilerHostUtil
    endif
    OBJ = o
    LIBEXT = a
    LIBPREFIX = lib
    BINEXT = 
endif

# Point to the necessary cross-compiler.

ifneq ($(TARGET_ARCH), $(HOST_ARCH))
    ifeq ($(TARGET_ARCH), aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/aarch64-unknown-nto-qnx7.0.0-g++
        endif
    endif

    ifdef HOST_COMPILER
        NVCC_COMPILER = -ccbin $(HOST_COMPILER)
    endif
endif

autoRangeSample: simplecuda.$(OBJ)
	nvcc $(NVCC_COMPILER) -o $@ $^ $(LIBS) $(INCLUDES)

simplecuda.$(OBJ): simplecuda.cu
	nvcc $(NVCC_COMPILER) -c $(INCLUDES) $<

run: autoRangeSample
	./$<

clean:
ifeq ($(OS), Windows_NT)
	del autoRangeSample.exe autoRangeSample.lib autoRangeSample.exp simplecuda.$(OBJ)
else
	rm -f autoRangeSample simplecuda.$(OBJ)
endif
