

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "backprop.h"
#include "omp.h"

extern char *strcpy();
extern void exit();

int layer_size = 0;
int num_iter_control = 0;
int dev_id = 0;
backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)
  
  printf("Input layer size : %d\n", layer_size);
  load(net);
  //entering the training kernel, only one iteration
  printf("Starting training kernel\n");
  bpnn_train_cuda(net, &out_err, &hid_err,num_iter_control,dev_id);
  bpnn_free(net);
  printf("Training done\n");
}

int setup(argc, argv)
int argc;
char *argv[];
{
	
  int seed;

  if (argc!=4){
  fprintf(stderr, "usage: backprop <num of input elements> <dev_id> <num of iter>\n");
  exit(0);
  }
  layer_size = atoi(argv[1]);
  if (layer_size%16!=0){
  fprintf(stderr, "The number of input points must be divided by 16\n");
  exit(0);
  }

  dev_id = atoi(argv[2]);
  num_iter_control = atoi(argv[3]);
  
  


  seed = 7;   
  bpnn_initialize(seed);
  backprop_face();

  exit(0);
}
