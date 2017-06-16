count=100
n=0
## while [ $n -lt $count ]; do
while true; do
##    watch -n 0.1 "nvidia-smi | awk 'NR==1||NR>=7&&NR<=12||NR>=18{print}'"
    nvidia-smi | awk 'NR==1||NR>=7&&NR<=13||NR>=18{print}'
    sleep 0.8
##    n=$(( $n + 1 ))
done
