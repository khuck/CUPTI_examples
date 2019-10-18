#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
# Cyclops
#CUDADIR=/usr/local/cuda-10.1
# Voltar
CUDADIR=/packages/cuda/9.2
CUPTIDIR=$(CUDADIR)/extras/CUPTI
INCLUDES=-I$(CUPTIDIR)/include

ifndef OS
 OS   := $(shell uname)
 HOST_ARCH := $(shell uname -m)
endif

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):../../libWin32:../../libx64
    LIBS= -lcuda -L ../../libWin32 -L ../../libx64 -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):../../lib
        LIBS= -Xlinker -framework -Xlinker cuda -L ../../lib -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(CUPTIDIR)/lib:$(CUPTIDIR)/lib64
        LIBS= -lcuda -L $(CUPTIDIR)/lib -L $(CUPTIDIR)/lib64 -lcupti
    endif
    OBJ = o
endif

all: activity_trace_async vec activity_trace_async.so

activity_trace_async: activity_trace_async.$(OBJ) vec.$(OBJ)
	nvcc -g -o $@ $^ $(LIBS) --default-stream per-thread -std=c++11 -Xcompiler="-pthread"

vec: vec.$(OBJ)
	nvcc -g -o $@ $^ $(LIBS) --default-stream per-thread -std=c++11 -Xcompiler="-pthread"

activity_trace_async.$(OBJ): activity_trace_async.cpp
	nvcc -g -c $(INCLUDES) $< -Xcompiler="-fPIC" -std=c++11 -Xcompiler="-pthread"

activity_trace_async.so: activity_trace_async.o
	nvcc -shared -g -o $@ $< $(LIBS) -std=c++11 -Xcompiler="-pthread"

vec.$(OBJ): vec.cu
	nvcc -g -c $(INCLUDES) $< --default-stream per-thread -std=c++11 -Xcompiler="-pthread"

run: activity_trace_async vec
	./$<

clean:
	rm -rf activity_trace_async activity_trace_async.$(OBJ) vec.$(OBJ) vec vec_clean.$(OBJ) traces*
