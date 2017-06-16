# kill gpu power sampling process
echo "waiting for power sampling to exit..."

while true; do
sleep 10
PID=`ps -aux | grep S[^+][^bash]*CodeXLPowerProfiler | awk '{print $2}'`
#echo ${PID}
#kill -9 $PID
#if [ ! $PID ]; then
if [ ! $PID ]; then
	echo "Power Sampling exits."
	sleep 5
	exit 0
	# echo $PID
	# echo `ps -aux | grep CodeXLPowerProfiler`
fi
done
