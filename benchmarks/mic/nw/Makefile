# C compiler
CC = icc 
ICC = icc
CC_FLAGS = -O3 -qopenmp
OFFLOAD_CC_FLAGS = -qoffload-option,mic,compiler,"-qno-opt-prefetch"

all: needle needle_offload

needle: 
	$(CC) $(CC_FLAGS) needle.cpp -o needle 

needle_offload:
	$(ICC) $(CC_FLAGS) $(OFFLOAD_CC_FLAGS) -DOMP_OFFLOAD needle.cpp -o needle_offload

clean:
	rm -f needle needle_offload
