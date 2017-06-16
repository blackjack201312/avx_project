# Add CodeXL tools path
export PATH=/opt/CodeXL_2.2-511:$PATH

SI=${SI:-500}
SD=${SD:-10}
OUTFILE=${OUTFILE:-$(pwd)/amd_pw_test}

CodeXLPowerProfiler -e all -o $OUTFILE -T $SI -d $SD -F txt
