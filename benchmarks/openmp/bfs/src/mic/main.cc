/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
/*
   Implementing Breadth first search on CUDA using algorithm given in DAC'10
   paper "An Effective GPU Implementation of Breadth-First Search"

   Copyright (c) 2010 University of Illinois at Urbana-Champaign. 
   All rights reserved.

   Permission to use, copy, modify and distribute this software and its documentation for 
   educational purpose is hereby granted without fee, provided that the above copyright 
   notice and this permission notice appear in all copies of this software and that you do 
   not sell the software.

   THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
   OTHERWISE.

Author: Lijiuan Luo (lluo3@uiuc.edu)
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <parboil.h>
#include "parboil.h"
#include <deque>
#include <iostream>
#include <omp.h>

#define MAX_THREADS_PER_BLOCK 512
#define INF 2147483647//2^31-1

#define UP_LIMIT 16677216//2^24
#define WHITE 16677217
#define GRAY 16677218
#define GRAY0 16677219
#define GRAY1 16677220
#define BLACK 16677221
int no_of_nodes; //the number of nodes in the graph
int edge_list_size;//the number of edges in the graph
FILE *fp;

//typedef int2 Node;
//typedef int2 Edge;

struct Node
{
	int x;
	int y;
};

struct Edge
{
	int x;
	int y;
};

const int h_top = 1;
const int zero = 0;

void runCPU(int argc, char** argv);
////////////////////////////////////////////////////////////////////
//the cpu version of bfs for speed comparison
//the text book version ("Introduction to Algorithms")
////////////////////////////////////////////////////////////////////
int* BFS_CPU( Node* h_graph_nodes, Edge* h_graph_edges, int source, int no_of_nodes, int edge_list_size)
{
	int *nodeX = (int*) malloc(sizeof(int)*no_of_nodes);
	int *nodeY = (int*) malloc(sizeof(int)*no_of_nodes);
	int *edgeX = (int*) malloc(sizeof(int)*edge_list_size);
	int *edgeY = (int*) malloc(sizeof(int)*edge_list_size);
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	int index;
	int cnt;
	for (cnt = 0; cnt < no_of_nodes; cnt++)
	{
		nodeX[cnt] = h_graph_nodes[cnt].x;
		nodeY[cnt] = h_graph_nodes[cnt].y;
		h_cost[cnt]=INF;
	}

	for (cnt = 0; cnt < edge_list_size; cnt++)
	{
		edgeX[cnt] = h_graph_edges[cnt].x;
		edgeY[cnt] = h_graph_edges[cnt].y;
	}

	h_cost[source] = 0;

        const char* env_iter = getenv("ITER");
        int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
	printf("[ITERATION NUM]:%d\n", iteration);

	double time = omp_get_wtime();
	#pragma offload target(mic:0)\
       	in(nodeX:length(no_of_nodes)) \
       	in(nodeY:length(no_of_nodes)) \
	in(edgeX:length(edge_list_size)) \
	in(edgeY:length(edge_list_size)) \
       	inout(h_cost:length(no_of_nodes)) \
	in(source, no_of_nodes, iteration) 
	{ 
		int iter;
		for(iter = 0; iter < iteration; iter++)
		{
			int *myDeque = (int*) malloc(sizeof(int) * no_of_nodes);
			int *head = myDeque;
			int *tail = myDeque;
			*tail = source;
			tail ++; 

			int j;
			int *h_color = (int*) malloc(sizeof(int)*no_of_nodes);
			for(j = 0; j < no_of_nodes; j++) h_color[j] = WHITE;
			h_color[0] = GRAY;
			while(tail - head > 0)
			{
				index = *head;
				*head = -1;
				head ++;
				#pragma omp parallel for shared(myDeque, head, tail)
				for(int i=nodeX[index]; i<(nodeY[index] + nodeX[index]); i++)
				{
					int id = edgeX[i]; // h_graph_edges[i].x;
					if(h_color[id] == WHITE)
					{
						h_cost[id] = h_cost[index] + 1;
						#pragma omp critical 
						{
							*tail = id;
							tail ++;
						}
						h_color[id] = GRAY;
					}
				}
				h_color[index] = BLACK;
			}
			free(myDeque);
			free(h_color);
		}
	}
	printf("[RUN TIME]:%f\n", omp_get_wtime() - time);
	return h_cost;
}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	runCPU(argc,argv);
}

///////////////////////////////
//FUNCTION: only run CPU version 
////////////////////////////////////////////
void runCPU( int argc, char** argv) 
{

	struct pb_Parameters *params;
	struct pb_TimerSet timers;

	pb_InitializeTimerSet(&timers);
	params = pb_ReadParameters(&argc, argv);
	if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
	{
		fprintf(stderr, "Expecting one input filename\n");
		exit(-1);
	}

	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	//printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(params->inpFiles[0],"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source;

	fscanf(fp,"%d",&no_of_nodes);
	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].x = start;
		h_graph_nodes[i].y = edgeno;
//		color[i]=WHITE;
	}
	//read the source node from the file
	fscanf(fp,"%d",&source);
	fscanf(fp,"%d",&edge_list_size);
	int id,cost;
	Edge* h_graph_edges = (Edge*) malloc(sizeof(Edge)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i].x = id;
		h_graph_edges[i].y = cost;
	}
	if(fp)
		fclose(fp);    

	unsigned int cpu_timer = 0;
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	int* h_cost = BFS_CPU( h_graph_nodes, h_graph_edges, source, no_of_nodes, edge_list_size);
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	if(params->outFile!=NULL)
	{
		//printf("Result stored in %s\n", params->outFile);
		FILE *fp = fopen(params->outFile,"w");
		fprintf(fp,"%d\n", no_of_nodes);
		for(int i=0;i<no_of_nodes;i++)
			fprintf(fp,"%d %d\n",i,h_cost[i]);
		fclose(fp);
	}


	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	// cleanup memory
	free(h_graph_nodes);
	free(h_graph_edges);
	free(h_cost);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);
	pb_PrintTimerSet(&timers);
	pb_FreeParameters(params);
}
