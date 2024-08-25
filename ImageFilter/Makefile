main: main.o filter.o
	@nvcc -o main *.o -g -m64 -I/usr/local/cuda/samples/Common `pkg-config opencv4 --cflags --libs` -gencode arch=compute_86,code=sm_86 --compiler-options -Wall -Xcompiler -fopenmp

filter.o:
	@nvcc -c -g --resource-usage filter.cu -I/usr/local/cuda/samples/Common `pkg-config opencv4 --cflags --libs`

main.o:
	@nvcc -c -g --resource-usage main.cu -I/usr/local/cuda/samples/Common `pkg-config opencv4 --cflags --libs`
