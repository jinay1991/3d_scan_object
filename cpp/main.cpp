#include "scan.hpp"

int main(int argc, char **argv)
{
    printf("usage: ./scan --input [image file]\n");
    if (argc < 2)
    {
        return -1;
    }
    char *filename;
    for (int i = 1; i < argc; i++)
    {
        if(!strcmp("--input", argv[i]))
        {
            filename = argv[i + 1];
        }
    }

    printf("filename: %s\n", filename);

    Scan scn(filename);
    scn.Video();

    return 0;
}