CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -fopenmp
LDLIBS ?= -lm

all: laplacian_benchmark

laplacian_benchmark: laplacian_benchmark.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f laplacian_benchmark

.PHONY: all clean
