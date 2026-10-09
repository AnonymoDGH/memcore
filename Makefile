CC = gcc
CFLAGS = -O3 -march=native -ffast-math -fopenmp -Wall -Wextra -std=c11 -Isrc
LDFLAGS = -lm -fopenmp

SRC = src/tensor.c src/model.c src/tasks.c src/memory.c src/infer.c src/train.c
OBJ = $(SRC:.c=.o)

all: memcore

memcore: src/main.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c src/*.h
	$(CC) $(CFLAGS) -c -o $@ $<

# gradient + kv-cache checks, built without fast-math
test: tests/gradcheck.c src/model.c src/tensor.c
	$(CC) -O1 -fopenmp -Wall -Wextra -std=c11 -Isrc -o gradcheck $^ -lm
	./gradcheck

clean:
	rm -f src/*.o memcore gradcheck

.PHONY: all test clean
