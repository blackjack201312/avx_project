all: rapl_lib_shared rapl_lib_static power_gov_static

rapl_lib_shared: 
	gcc -fpic -c -g msr.c cpuid.c rapl.c 
	gcc -shared -o librapl.so msr.o cpuid.o rapl.o

rapl_lib_static: 
	gcc -c -g msr.c cpuid.c rapl.c 
	ar rcs librapl.a msr.o cpuid.o rapl.o

power_gov_static: 
	gcc power_gov.c -I. -L. -o power_gov ./librapl.a -lm

power_gov: 
	gcc power_gov.c -I. -L. -lrapl -o power_gov -lm

clean: 
	rm -f power_gov librapl.so librapl.a msr.o cpuid.o rapl.o 
