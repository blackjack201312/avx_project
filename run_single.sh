export OMP_NUM_THREADS=40
#platform=openmp dev=cpu workload=normal tool=sgemm ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=mri-q ./run_template.sh
platform=openmp dev=cpu workload=normal tool=spmv ./run_template.sh
