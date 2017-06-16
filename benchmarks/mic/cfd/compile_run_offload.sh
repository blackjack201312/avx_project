make clean
export OMP_NUM_THREADS=56
export MIC_OMP_NUM_THREADS=56
export ITER=10000
make
echo
echo "====RUN===="
./euler3d_cpu_offload ../../data/cfd/fvcorr.domn.193K

make clean
export OMP_NUM_THREADS=112
export MIC_OMP_NUM_THREADS=112
export ITER=13000
make
echo
echo "====RUN===="
./euler3d_cpu_offload ../../data/cfd/fvcorr.domn.193K

make clean
export OMP_NUM_THREADS=224
export MIC_OMP_NUM_THREADS=224
export ITER=18000
make
echo
echo "====RUN===="
./euler3d_cpu_offload ../../data/cfd/fvcorr.domn.193K
