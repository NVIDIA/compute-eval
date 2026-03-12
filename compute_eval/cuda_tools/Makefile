# Makefile for building libcuda_profile.so locally.
#
# The canonical build logic lives in hatch_build.py (used during wheel
# builds).  Keep this file in sync with it.
#
# Environment variables (all optional):
#   CXX          - C++ compiler          (default: g++)
#   CUDA_HOME    - CUDA toolkit root     (default: /usr/local/cuda)
#   CUPTI_INCLUDE - CUPTI header dir     (auto-detected from CUDA_HOME)
#   CUPTI_LIB    - CUPTI library dir     (auto-detected from CUDA_HOME)

CXX          ?= g++
CUDA_HOME    ?= /usr/local/cuda

# Auto-detect architecture
UNAME_M      := $(shell uname -m)
ifeq ($(filter $(UNAME_M),aarch64 arm64),)
  CUDA_TARGET := x86_64-linux
else
  CUDA_TARGET := sbsa-linux
endif

# CUPTI paths — match the fallback logic in hatch_build.py
CUPTI_INCLUDE ?= $(firstword $(wildcard \
    $(CUDA_HOME)/targets/$(CUDA_TARGET)/include) \
    $(CUDA_HOME)/include)
CUPTI_LIB     ?= $(firstword $(wildcard \
    $(CUDA_HOME)/lib64) \
    $(CUDA_HOME)/targets/$(CUDA_TARGET)/lib)

CXXFLAGS := -std=c++14 -fPIC -shared
LDFLAGS  := -Wl,--no-as-needed -Wl,--disable-new-dtags -Wl,-rpath,$(CUPTI_LIB)
LIBS     := -lcupti -ldl

SOURCES := \
    libcuda_profile.cpp \
    nvtx_interpose.cpp \
    profiler_state.cpp \
    nvtx_correlation.cpp \
    activity_processing.cpp \
    json_output.cpp \
    profiler_lifecycle.cpp

TARGET := libcuda_profile.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(wildcard *.h)
	$(CXX) $(CXXFLAGS) -I$(CUPTI_INCLUDE) -L$(CUPTI_LIB) $(LDFLAGS) $(LIBS) -o $@ $(SOURCES)

clean:
	$(RM) $(TARGET)
