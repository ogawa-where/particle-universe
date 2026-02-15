NVCC = nvcc
NVCC_FLAGS = -shared -Xcompiler -fPIC -O3 -arch=sm_86
TARGET = libparticle_sim.so
SRC = particle_sim.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC) -lcurand

clean:
	rm -f $(TARGET)

run: $(TARGET)
	python3 server.py

.PHONY: all clean run
