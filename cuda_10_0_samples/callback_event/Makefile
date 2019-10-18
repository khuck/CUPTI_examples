#
# Copyright 2011-2015 NVIDIA Corporation. All rights reserved
# 
INCLUDES=-I$(CUDA)/include -I$(CUDA)/extras/CUPTI/include

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
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(CUDA)/lib
        LIBS= -Xlinker -framework -Xlinker cuda -L ../../lib -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(CUDA)/lib:$(CUDA)/lib64:$(CUDA)/extras/CUPTI/lib64
        LIBS= -lcuda -L $(CUDA)/lib -L $(CUDA)/lib64 -L $(CUDA)/extras/CUPTI/lib64 -lcupti
    endif
    OBJ = o
endif

callback_event: callback_event.$(OBJ)
	nvcc -o $@ $(flags) callback_event.$(OBJ) $(LIBS)

callback_event.$(OBJ): callback_event.cu
	nvcc $(flags) -c $(INCLUDES) $<

run: callback_event
	./$<

clean:
	rm -f callback_event callback_event.$(OBJ)
