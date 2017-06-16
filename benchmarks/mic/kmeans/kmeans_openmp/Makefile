# C compiler
CC = icc
CC_FLAGS = -qopenmp -O3 -fp-model fast=2 -fma -qoffload-option,mic,compiler,"-qno-opt-prefetch"

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o  -o kmeans

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c
	
getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c

clean:
	rm -f *.o *~ kmeans 
