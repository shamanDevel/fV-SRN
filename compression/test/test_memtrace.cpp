#include "../src/memtrace.h"

#include <vector>

void testMemtraceFunction()
{
    const int n = 5;
    for (int i=0; i<n;++i)
    {
        std::vector<char> v1(1000);
        v1[5] = 4;

        char* v2 = new char[100];
        v2[5] = 3;

        char* v3 = static_cast<char*>(malloc(10));
        v3[2] = 2;

        free(v3);
        delete[] v2;
    }
}