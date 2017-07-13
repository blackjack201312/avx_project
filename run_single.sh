export OMP_NUM_THREADS=40
#platform=openmp dev=cpu workload=normal tool=sgemm ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=mri-q ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=spmv ./run_template.sh
platform=openmp dev=cpu workload=normal tool=nn ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=lud ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=kmeans ./run_template.sh

#platform=openmp dev=cpu workload=normal tool=backprop ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=bfs ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=cfd ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=cutcp ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=histo ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=hotspot ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=nw ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=pathfinder ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=srad ./run_template.sh
#platform=openmp dev=cpu workload=normal tool=stencil ./run_template.sh
