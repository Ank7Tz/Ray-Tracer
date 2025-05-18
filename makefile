PROJECT = Ray-Tracer

NVCC = nvcc
NVCCFLAGS =-g -G

GCC = g++
GCCFLAGS =-g

BUILD_DIR = build
SRC_DIR = src
SERVER_DIR = server

SOURCES = $(SRC_DIR)/main.cu
TARGET = $(BUILD_DIR)/ray_tracer
SERVER_TARGET = $(BUILD_DIR)/server
SERVER_SOURCE = $(SERVER_DIR)/server.cpp

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

clean:
	rm -r $(BUILD_DIR)

rebuild:
	clean all

server: directories $(SERVER_TARGET)

$(SERVER_TARGET): $(SERVER_SOURCE)
	$(GCC) $(GCCFLAGS) -o $@ $^

.PHONY: all directories clean rebuild