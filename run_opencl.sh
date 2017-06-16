platform=opencl
dev=amd
workload=light

tool=bfs ./run_single.sh
tool=cutcp ./run_single.sh
tool=histo ./run_single.sh
tool=sgemm ./run_single.sh
tool=spmv ./run_single.sh
tool=stencil ./run_single.sh
tool=mri-q ./run_single.sh
tool=backprop ./run_single.sh
tool=hotspot ./run_single.sh
tool=pathfinder ./run_single.sh
tool=lud ./run_single.sh
tool=nw ./run_single.sh
tool=kmeans ./run_single.sh
tool=srad ./run_single.sh
tool=nn ./run_single.sh
tool=cfd ./run_single.sh


