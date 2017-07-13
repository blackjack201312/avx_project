/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */
/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/

/*************************************************************************/
/**   File:         example.c                                           **/
/**   Description:  Takes as input a file:                              **/
/**                 ascii  file: containing 1 data point per line       **/
/**                 binary file: first int is the number of objects     **/
/**                              2nd int is the no. of features of each **/
/**                              object                                 **/
/**                 This example performs a fuzzy c-means clustering    **/
/**                 on the data. Fuzzy clustering is performed using    **/
/**                 min to max clusters and the clustering that gets    **/
/**                 the best score according to a compactness and       **/
/**                 separation criterion are returned.                  **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department Northwestern University                   **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <malloc.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>
#include "getopt.h"

#include "kmeans.h"

extern double wtime(void);

int num_omp_threads = 40;

/*---< usage() >------------------------------------------------------------*/
void usage(char *argv0) {
  char *help =
  "Usage: %s [switches] -i filename\n"
  "       -i filename     		: file containing data to be clustered\n"
  "       -b                 	: input file is in binary format\n"
  "       -k                 	: number of clusters (default is 5) \n"
  "       -t threshold		: threshold value\n"
  "       -n no. of threads	: number of threads\n";
  fprintf(stderr, help, argv0);
  exit(-1);
}

/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
  int     opt;
  extern char   *optarg;
  extern int     optind;
  int     nclusters=5;
  char   *filename = 0;
  float  *buf;
  float **attributes;
  float **cluster_centres=NULL;
  int     i, j;

  int     numAttributes;
  int     numObjects;
  char    line[1024];
  int     isBinaryFile = 0;
  int     nloops = 1;
  float   threshold = 0.001;
  double  timing;

  while ( (opt=getopt(argc,argv,"i:k:t:b:n:?"))!= EOF) {
    switch (opt) {
      case 'i': filename=optarg;
      break;
      case 'b': isBinaryFile = 1;
      break;
      case 't': threshold=atof(optarg);
      break;
      case 'k': nclusters = atoi(optarg);
      break;
      case 'n': num_omp_threads = atoi(optarg);
      break;
      case '?': usage(argv[0]);
      break;
      default: usage(argv[0]);
      break;
    }
    //printf("num_omp_threads=%d\n", num_omp_threads);

  }


  if (filename == 0) usage(argv[0]);

  numAttributes = numObjects = 0;

  /* from the input file, get the numAttributes and numObjects ------------*/

  if (isBinaryFile) {
    int infile;
    if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    printf("INFILE: %d", infile);
    read(infile, &numObjects,    sizeof(int));
    read(infile, &numAttributes, sizeof(int));


    /* allocate space for attributes[] and read attributes of all objects */
    buf           = (float*) memalign(16, numObjects*numAttributes*sizeof(float));
    attributes    = (float**)memalign(16, numObjects*             sizeof(float*));
    attributes[0] = (float*) memalign(16, numObjects*numAttributes*sizeof(float));
    for (i=1; i<numObjects; i++)
    attributes[i] = attributes[i-1] + numAttributes;

    read(infile, buf, numObjects*numAttributes*sizeof(float));

    close(infile);
  }
  else {
    FILE *infile;
    if ((infile = fopen(filename, "r")) == NULL) {
      fprintf(stderr, "Error: no such file (%s)\n", filename);
      exit(1);
    }
    while (fgets(line, 1024, infile) != NULL)
    if (strtok(line, " \t\n") != 0)
    numObjects++;
    rewind(infile);
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") != 0) {
        /* ignore the id (first attribute): numAttributes = 1; */
        while (strtok(NULL, " ,\t\n") != NULL) {
          //          printf("NULL != NULL!!!\n");
          numAttributes++;
        }
        break;
      }
    }


    /* Do padding for AVX*/
    int realAttr;
    realAttr = numAttributes;
    numAttributes += (8-(numAttributes%8));
    /* allocate space for attributes[] and read attributes of all objects */
    buf           = (float*) memalign(16, numObjects*numAttributes*sizeof(float));
    attributes    = (float**)memalign(16, numObjects*             sizeof(float*));
    attributes[0] = (float*) memalign(16, numObjects*numAttributes*sizeof(float));
    for (i=1; i<numObjects; i++)
    attributes[i] = attributes[i-1] + numAttributes;
    rewind(infile);
    i = 0;
    while (fgets(line, 1024, infile) != NULL) {
      if (strtok(line, " \t\n") == NULL) continue;
      for (j=0; j<numAttributes; j++) {
        if(j < realAttr)
        buf[i] = atof(strtok(NULL, " ,\t\n"));
        else
        buf[i] = 0;
        printf("buf[i]:%f, i:%d\n", buf[i], i);
        i++;
      }
    }
    fclose(infile);
  }
  printf("I/O completed\n");

  memcpy(attributes[0], buf, numObjects*numAttributes*sizeof(float));

  const char* env_itrs = getenv("ITERS");
  int nIter = (env_itrs != NULL) ? atoi(env_itrs) : 1;
  const char* env_secs = getenv("SECS");
  int secs = (env_secs != NULL) ? atoi(env_secs) : -1;
  //int timeRestrict = (env_secs != NULL) ? 1 : 0;

  double start_t = 0;
  double end_t = 0;
  double total_s0, total_s1, total_s2 = 0.0;
  //double total_s0, total_s1, total_s2;
  //total_s0 = total_s1 = total_s2 = 0;
  int c0, c1, c2;
  c0 = 0;
  c1 = 0;
  c2 = 0;

  if(secs == -1) {
    fprintf(stderr, "You must set a time larger than 0 seconds!\n");
    return 1;
  }

  printf("[Number of threads]:%d\n", num_omp_threads);

  for (int i = -60000; i<nIter; i++) {
    start_t = omp_get_wtime();
    //printf("%lf\n", start_t);
    if(i == 0)
    timing = omp_get_wtime();

    cluster_centres = NULL;
    //printf("num_omp_threads = %d\n", num_omp_threads);
    cluster(numObjects,
      numAttributes,
      attributes,           /* [numObjects][numAttributes] */
      nclusters,
      threshold,
      &cluster_centres,
      num_omp_threads
    );
    break;
    end_t = omp_get_wtime();
    if(i < -40000)
    {
      c0++;
      total_s0 += end_t - start_t;
    }
    else if(i < -20000 && i >= -40000)
    {
      c1++;
      total_s1 += end_t - start_t;
    }
    else if(i >= -20000)
    {
      c2++;
      total_s2 += end_t - start_t;

      if (i == -1)
      {
        double tPerIter = total_s2 / c2;
        printf("Sampling from the first 20000 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s0 / (double)c0, (int)((double)secs / (total_s0 / (double)c0)));
        printf("Sampling from the middle 20000 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s1 / (double)c1, (int)((double)secs / (total_s1 / (double)c1)));
        printf("Sampling from the last 20000 itrs.\nEstimated time: %lf s.\n", tPerIter);
        nIter = (int)((double)secs / tPerIter) + 1;
        printf("Adjust %d iterations to meet %d seconds.\n", nIter, secs);
        c2 = 0;
        total_s2=0;
      }
    }
  }

  timing = omp_get_wtime() - timing;

  double averMsecs = total_s2 / nIter * 1000;
  printf("iterated %d times, average time is %lf ms.\n", nIter, averMsecs);

  printf("number of Clusters %d\n",nclusters);
  printf("number of Attributes %d\n\n",numAttributes);
  /*  	printf("Cluster Centers Output\n");
  printf("The first number is cluster number and the following data is arribute value\n");
  printf("=============================================================================\n\n");

  for (i=0; i< nclusters; i++) {
  printf("%d: ", i);
  for (j=0; j<numAttributes; j++)
  printf("%.2f ", cluster_centres[i][j]);
  printf("\n\n");
}
*/
printf("Time for process: %f\n", timing);

free(attributes);
free(cluster_centres[0]);
free(cluster_centres);
free(buf);
return(0);
}
