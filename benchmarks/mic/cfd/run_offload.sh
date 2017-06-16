make clean
export OMP_NUM_THREADS=$OMP_NUM_THREADS
export ITER=$ITER
export MIC_OMP_NUM_THREADS=$OMP_NUM_THREADS
make
./euler3d_cpu_offload ../../data/cfd/fvcorr.domn.193K
