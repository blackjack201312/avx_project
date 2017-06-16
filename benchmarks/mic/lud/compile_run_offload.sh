make clean

make

export OMP_NUM_THREADS=224
export ITER=80
./omp/lud_omp_offload -s 8000 

export OMP_NUM_THREADS=112
export ITER=60
./omp/lud_omp_offload -s 8000 

export OMP_NUM_THREADS=56
export ITER=35
./omp/lud_omp_offload -s 8000 
