# kill gpu power sampling process
# PID=`ps -aux | grep S[^+]*[^sudo]*power_gov | awk '{print $2}'`
PID=`ps -aux | grep ./cpu_power/power_gov | awk '{print $2}'`
echo ${PID}
sudo kill -9 $PID
