#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

void run(int argc, char** argv);

/* define timer macros */
double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

#define BENCH_PRINT

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9

void
init(int argc, char** argv)
{
	if(argc==3){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
	}else{
                printf("Usage: pathfiner width num_of_steps\n");
                exit(0);
        }
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];
// #ifdef BENCH_PRINT
//    for (int i = 0; i < rows; i++)
//    {
//        for (int j = 0; j < cols; j++)
//        {
//            printf("%d ",wall[i][j]) ;
//        }
//        printf("\n") ;
//    }
// #endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    double cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    int i;
    const char* env_iter = getenv("ITER");
    int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
    printf("[ITERATION NUM]:%d\n", iteration);

    double start = gettime();
    for (i = 0 ; i<iteration;i++)
    for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
        #pragma omp parallel for private(min)
        for(int n = 0; n < cols; n++){
          min = src[n];
          if (n > 0)
            min = MIN(min, src[n-1]);
          if (n < cols-1)
            min = MIN(min, src[n+1]);
          dst[n] = wall[t+1][n]+min;
        }
    }

    double end = gettime();
    printf("Finish %d iterations, spent %lf secs!\n", iteration, end - start);

    delete [] data;
    delete [] wall;
    delete [] dst;
    delete [] src;
}

