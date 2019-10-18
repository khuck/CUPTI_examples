#
# Copyright 2015 NVIDIA Corporation. All rights reserved
#

# CUPTI OpenACC support is only available on Linux x86_64 and ppc64le.

# Point to your PGI OpenACC installation's include directory.
# PGI compiler must be in PATH.
OPENACC_INCLUDE_PATH =

HOST_ARCH := $(shell uname -m)

INCLUDES += -I../../include
INCLUDES += -I../../../../include
INCLUDES += -I$(OPENACC_INCLUDE_PATH)

PGCPP       = pgc++
PGCPP_FLAGS = -acc -ta=nvidia:cuda7.0 -Mcuda=nordc -pgf90libs
ifneq ($(HOST_ARCH),ppc64le)
        PGCPP_FLAGS += -tp=p7 -mp=nonuma
else
        PGCPP_FLAGS += -DHOST_ARCH_PPC=1
endif

export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):../../lib64
TRACE_LIB = libopenacc_trace.so

all: openacc_app

$(TRACE_LIB): openacc_trace.cpp
	$(PGCPP) $(INCLUDES) -DCUPTI_OPENACC_SUPPORT -fPIC -shared -o $@ $<

openacc_app: $(TRACE_LIB) openacc_app.cpp
	$(PGCPP) $(INCLUDES) $(PGCPP_FLAGS)         \
	-L. -L../../lib64  -lopenacc_trace -lcupti  \
	-o $@ $^

run: $(TRACE_LIB) openacc_app
	ACC_PROFLIB=$(TRACE_LIB) ./openacc_app

clean:
	rm -f $(TRACE_LIB) openacc_app

