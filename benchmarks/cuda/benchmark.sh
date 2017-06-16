# Source intel environment
# source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

sleep_interval=60

nohup ./gpu_pw_sampling.sh 1>power.log 2>&1 &
echo "waiting 60 seconds."
sleep ${sleep_interval}

# backprop
cd ./backprop
make clean
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=600000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=600000 ./run

cd ..

# hotspot
cd ./hotspot
make clean
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=1600000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=1600000 ./run

cd ..

# pathfinder
cd ./pathfinder
make clean 
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=80000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=80000 ./run

cd ..

# lud
cd ./lud
make clean 
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=400 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=400 ./run

cd ..

# nw
cd ./nw
make clean 
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=120000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=120000 ./run

cd ..

# kmeans
cd ./kmeans
make clean 
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=120000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=120000 ./run

cd ..

# srad_v2
cd ./srad/srad_v2
make clean
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=25000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=25000 ./run

cd ../..

# nn
cd ./nn
make clean
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=10000000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=10000000 ./run

cd ..

# cfd
cd ./cfd
make clean
make

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=0 ITER=25000 ./run

echo "waiting 60 seconds."
sleep ${sleep_interval}
DEV_ID=1 ITER=25000 ./run

cd ..

echo "sleep 60 seconds."
sleep ${sleep_interval}
./kill_gpu_pw_sampling.sh
# Deactivate intel environment
# source ~/.bashrc
