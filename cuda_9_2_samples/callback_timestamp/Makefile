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

callback_timestamp: callback_timestamp.$(OBJ)
	nvcc -o $@ callback_timestamp.$(OBJ) $(LIBS)

callback_timestamp.$(OBJ): callback_timestamp.cu
	nvcc -c $(INCLUDES) $<

run: callback_timestamp
	./$<

clean:
	rm -f callback_timestamp callback_timestamp.$(OBJ)
