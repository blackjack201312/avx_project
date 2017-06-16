
## Introduction

EPPMiner is a benchmark suites to evaluate the performance, power and energy characterizations of different heterogeneous systems. Please refer to http://eppminer.comp.hkbu.edu.hk/ for your testing results and more details. The project will keep updated actively with more programs and functions.

## Functionality

1. Support three parallel programming techniques
2. Support several processors and accelerators
3. Flexible workload settings

## Run experiments

1. To run a single benchmark application, one should indicate the parallel platform name, the device and the workload. For example, we can run sgemm with OpenCL on AMD GPU with the following command:

```
platform=opencl dev=amd workload=normal tool=sgemm ./run_single.sh
```

2. You can config global settings in file run_single.sh or give in runtime.
3. run_opencl.sh gives a batch running example for AMD GPU on the whole benchmark suites.
