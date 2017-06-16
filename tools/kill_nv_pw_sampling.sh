# kill gpu power sampling process
PID=`ps -aux | grep S[^+]*gpu_pw_sampling.sh | awk '{print $2}'`
echo ${PID}
kill -9 $PID
