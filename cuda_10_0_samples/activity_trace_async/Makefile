#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I../../include

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
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):../../lib:../../lib64
        LIBS= -lcuda -L ../../lib -L ../../lib64 -lcupti
    endif
    OBJ = o
endif

activity_trace_async: activity_trace_async.$(OBJ) vec.$(OBJ)
	nvcc -o $@ $^ $(LIBS)

activity_trace_async.$(OBJ): activity_trace_async.cpp
	nvcc -c $(INCLUDES) $<

vec.$(OBJ): vec.cu
	nvcc -c $(INCLUDES) $<

run: activity_trace_async
	./$<

clean:
	rm -f activity_trace_async activity_trace_async.$(OBJ) vec.$(OBJ)
