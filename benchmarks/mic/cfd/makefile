all: euler3d_cpu_offload

euler3d_cpu_offload:
	icc -g -O3 -DOMP_OFFLOAD -Dblock_length=$(OMP_NUM_THREADS) -qopenmp -qoffload-option,mic,compiler,"-qno-opt-prefetch" euler3d_cpu.cpp -o euler3d_cpu_offload

clean:
	rm -f euler3d_cpu_offload
