#include<stdio.h>
#include<stdlib.h>
#include<string.h>

int main(int argc, char** argv){
    char str[] = "I am not a pig, heng~";
    char *strs;
    strs = strtok(str, " ,\t\n");
        printf("%s\n", strs);
    while(strs = strtok(NULL, " ,\n\t")){
        printf("%s\n", strs);
       //strcpy(strs, strtok(NULL, " ,\t\n"));
    }
    if (NULL){
        printf("NULL is true\n");

    }
    return 0;
}
