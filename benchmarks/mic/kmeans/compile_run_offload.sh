make clean
make
export OMP_NUM_THREADS=224
export ITER=14500
./kmeans_openmp/kmeans -n $OMP_NUM_THREADS -i ../../data/kmeans/kdd_cup 

sleep 10
export OMP_NUM_THREADS=112
export ITER=11250
./kmeans_openmp/kmeans -n $OMP_NUM_THREADS -i ../../data/kmeans/kdd_cup 

sleep 10
export OMP_NUM_THREADS=56
export ITER=6125
./kmeans_openmp/kmeans -n $OMP_NUM_THREADS -i ../../data/kmeans/kdd_cup 
