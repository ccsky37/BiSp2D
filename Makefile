NVCC ?= nvcc
ARCH ?= 80

TARGET = build/bisp2d
SOURCES = src/main.cu src/bisp2d.cu

all: $(TARGET)

$(TARGET): $(SOURCES) include/bisp2d.cuh
	mkdir -p build
	$(NVCC) -O3 -std=c++17 -arch=sm_$(ARCH) -Iinclude $(SOURCES) -o $(TARGET)

clean:
	rm -rf build

.PHONY: all clean
