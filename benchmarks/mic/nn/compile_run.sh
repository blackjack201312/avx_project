make clean

make

export OMP_NUM_THREADS=224
export ITER=36000
./nn filelist 10 30 90

export OMP_NUM_THREADS=112
export ITER=22000
./nn filelist 10 30 90

export OMP_NUM_THREADS=56
export ITER=18000
./nn filelist 10 30 90
