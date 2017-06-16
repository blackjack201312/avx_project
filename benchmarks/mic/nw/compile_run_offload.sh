make clean
make

export OMP_NUM_THREADS=224
export ITER=9000
./needle_offload 2048 10 $OMP_NUM_THREADS

export OMP_NUM_THREADS=112
export ITER=7000
./needle_offload 2048 10 $OMP_NUM_THREADS

export OMP_NUM_THREADS=56
export ITER=5500
./needle_offload 2048 10 $OMP_NUM_THREADS
