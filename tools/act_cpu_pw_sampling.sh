SI=${SI:-500}
OUTFILE=${OUTFILE:-$(pwd)/cpu_pw_test}

sudo ./cpu_power/power_gov -e $SI > $OUTFILE
