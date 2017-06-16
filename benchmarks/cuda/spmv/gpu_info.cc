#include <endian.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdio.h>
#include <inttypes.h>

void compute_active_thread(unsigned int *thread,
					unsigned int *grid,
					int task,
					int pad,
					int major,
					int minor,
					int warp_size,
					int sm)
{
	int max_thread;
	int max_warp;
	int max_block=8;
	printf("minor vers: %d, major vers: %d.\n", minor, major);
	if(major==1)
	{
		if(minor>=2)
		{
			max_thread=1024;
			max_warp=32;
		}
		else
		{
			max_thread=768;
			max_warp=24;
		}
	}
	else if(major==2)
	{
		max_thread=1536;
		max_warp=48;
	}
	else if(major==3)
	{
		//newer GPU  //keep using 2.0
		max_thread=2048;
		max_warp=64;
	}
	else if(major==5)
	{
		// Maxwell GPU
		max_thread=2048;
		max_warp=64;
	}
	else if(major==6)
	{
		// Pascal GPU
		max_thread=2048;
		max_warp=64;
	}
	
	int _grid;
	int _thread;
	int threads_per_sm=0;
	if(task*pad>sm*max_thread)
	{
		//_grid=sm*max_block;
		_thread=max_thread/max_block;
		_grid=(task*pad+_thread-1)/_thread;
	}
	else
	{
		_thread=pad;
		_grid=task;
	}
	thread[0]=_thread;
	grid[0]=_grid;
	
}
