#include <stdio.h>
#include <stdlib.h>

int main()
{
	printf("Get environment variable...\n");
	const char* s = getenv("ITERATION");
	printf("As string ITERATION=%s\n", (s != NULL) ? s : "No environment variable find...");
	int s_int = atoi(s);
	printf("As int ITERATION=%d\n", s_int);
	printf("End test...\n");
}
