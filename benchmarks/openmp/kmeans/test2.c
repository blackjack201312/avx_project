#include<stdio.h>

int main(){
    float a, b, c;
    a = 625.0;
    b = 147646800.0;
    c = a+b;
    printf("c=%f\tc-a=%f\tc-b=%f\n", c, c-a, c-b);
    return 0;
}
