#bin/bash

# benchmark configuration
CWD=$(pwd)
tool="${tool:-bfs}"
platform="${platform:-opencl}"
dev="${dev:-amd}"
workload="${workload:-light}"

toolpath=$CWD/benchmarks/$platform/$tool
dataconfig=$CWD/datasets/$tool/$workload/data.config.mk
datapath=$CWD/datasets/$tool/$workload
LOG_PATH=$CWD/logs
TOOL_PATH=$CWD/tools

# running time/power sampling frequency control
SLEEP_SECS=2
RUN_SECS=10
PW_SAMPLE_INT=500
PW_SAMPLE_SECS=5

# start power sampling
# echo Now come to $tool...
# echo start amd power sampling...
# export SI=$PW_SAMPLE_INT; export SD=$PW_SAMPLE_SECS; export OUTFILE=$LOG_PATH/benchmark_${tool}_${platform}_${dev}_${workload}_pw.log; nohup sh $TOOL_PATH/act_${dev}_pw_sampling.sh 1>/dev/null 2>&1 &

# compiling
cd $toolpath
make clean
echo compile...
make 1>compile.log 2>&1
echo sleep $SLEEP_SECS seconds...
sleep $SLEEP_SECS

# running
source $dataconfig
echo "start running application..."
echo $in_file
echo $out_file
SECS=${RUN_SECS} ./$tool $in_file $out_file > $LOG_PATH/benchmark_${tool}_${platform}_${dev}_${workload}_perf.log
echo end running application...
echo sleep $SLEEP_SECS seconds...
sleep $SLEEP_SECS

# kill power sampling
# cd $CWD
# echo "stop amd power sampling..."
# sh $TOOL_PATH/kill_${dev}_pw_sampling.sh

