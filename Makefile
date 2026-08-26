CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -Isrc
LDFLAGS = -lm

SRC = src/tensor.c src/tokenizer.c src/attention.c src/neural_mem.c \
      src/replay.c src/train.c
OBJ = $(SRC:.c=.o)

all: memcore

memcore: src/main.o $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: memcore_debug
	./memcore_debug

memcore_debug: src/main.o $(SRC) config/test.c
	$(CC) -g -O0 -Wall -Wextra -std=c11 -Isrc -Iconfig \
		-o $@ src/main.o $(filter-out src/main.o,$^) $(LDFLAGS)

clean:
	rm -f *.o src/*.o memcore memcore_debug

.PHONY: all test clean
