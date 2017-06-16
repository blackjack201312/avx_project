
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>


/*--------- using gettimeofday ------------*/

#include <sys/time.h>

struct timeval starttime;
struct timeval endtime;

#define startTime() \
{ \
  gettimeofday(&starttime, 0); \
}
#define stopTime(valusecs) \
{ \
  gettimeofday(&endtime, 0); \
  valusecs = (endtime.tv_sec-starttime.tv_sec)*1000000+endtime.tv_usec-starttime.tv_usec; \
}
