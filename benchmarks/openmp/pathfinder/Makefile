CC = icpc
SRC = pathfinder.cpp
EXE = pathfinder
FLAGS = -g -qopenmp -O2 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread

release:
	$(CC) $(SRC) $(FLAGS) -o $(EXE)

debug:
	$(CC) $(SRC) -g -Wall -o $(EXE)

clean:
	rm -f pathfinder

test:
	./pathfinder 100000 100

