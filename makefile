PROJECT = Ray-Tracer

CXX = nvcc
CXXFLAGS =

BUILD_DIR = build
SRC_DIR = src

SOURCES = $(SRC_DIR)/main.cu
TARGET = $(BUILD_DIR)/ray_tracer

all: directories $(TARGET)

directories:
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -r $(BUILD_DIR)

rebuild:
	clean all

.PHONY: all directories clean rebuild