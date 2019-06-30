// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cufft.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
// Adds an additional library so that timeGetTime() can be used

#include <stdlib.h>

#include <time.h>
#include <omp.h>



typedef float2 Complex;
/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/

#define NX 256
#define BATCH 1



static __global__ void ComplexPointwiseMulAndScale(Complex* A, Complex* B, double* res, int sizeA, int sizeB)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	double valueA = 0.0;
	double valueB = 0.0;
	double diff = 0.0;
	//double fab[sizeA];
	
	//int i = threadID;

	for (int i = threadID; i < ((sizeA- sizeB)+1); i += numThreads) {
		if ((i + (sizeB - 1)) < sizeA) {
			res[i] = 0.0;
			for (int j = 0; j < sizeB; j++) {
					//valueA = sqrt(pow(A[i + j].x, 2) + pow(A[i + j].y, 2));
					//valueB = sqrt(pow(B[i].x, 2) + pow(B[i].y, 2));
					res[i] += fabs(sqrt(pow(A[i + j].x, 2) + pow(A[i + j].y, 2)) - sqrt(pow(B[i].x, 2) + pow(B[i].y, 2)));
					//res[1] += fab;
				
			}

		}
	}
	//res[1] = fab;
}

int PadData(const Complex* signal, Complex** padded_signal, int signal_size,
	const Complex* filter_kernel, Complex** padded_filter_kernel, int filter_kernel_size)
{
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	// Pad signal
	Complex* new_data = (Complex*)malloc(sizeof(Complex) * new_size);
	memcpy(new_data + 0, signal, signal_size * sizeof(Complex));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(Complex));
	*padded_signal = new_data;

	// Pad filter
	new_data = (Complex*)malloc(sizeof(Complex) * new_size);
	memcpy(new_data + 0, filter_kernel + minRadius, maxRadius * sizeof(Complex));
	memset(new_data + maxRadius, 0, (new_size - filter_kernel_size) * sizeof(Complex));
	memcpy(new_data + new_size - minRadius, filter_kernel, minRadius * sizeof(Complex));
	*padded_filter_kernel = new_data;

	return new_size;
}




int readFile(int **grades, char *addr);







int CompareWav()
{
	
	printf("[simpleCUFFT] is starting...\n");

	int *h_A_real = NULL;
	unsigned int count_A;
	count_A = readFile(&h_A_real, "M1.txt");


	int *h_B_real = NULL;
	unsigned int count_B;
	count_B = readFile(&h_B_real, "M2.txt");




	Complex* h_A = (Complex*)malloc(sizeof(Complex) * count_A);
	// Initalize the memory for the signal
	for (unsigned int i = 0; i < count_A; ++i) {
		printf("Int is %d \n", h_A_real[i]);
		h_A[i].x = h_A_real[i] + 0.0;
		printf("Num is %f \n", h_A[i].x);
		h_A[i].y = 0;
	}

	Complex* h_B = (Complex*)malloc(sizeof(Complex) * count_B);
	// Initalize the memory for the signal
	for (unsigned int i = 0; i < count_B; ++i) {
		printf("Int is %d \n", h_B_real[i]);
		h_B[i].x = h_B_real[i] + +0.0;;
		printf("Num is %f \n", h_B[i].x);
		h_B[i].y = 0;
	}

	/*

	// Pad signal and filter kernel
	Complex* h_padded_signal;
	Complex* h_padded_filter_kernel;
	int new_size = PadData(h_signal, &h_padded_signal, SIGNAL_SIZE,
		h_filter_kernel, &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
	int mem_size = sizeof(Complex) * new_size;

	*/

	int MinCount = count_B;

	if (count_A < count_B) {
		MinCount = count_A;
	}

	unsigned int MaxCount = count_B;

	if (count_A > count_B) {
		MaxCount = count_A;
	}

	unsigned int size_A = sizeof(Complex)* count_A;
	unsigned int size_B = sizeof(Complex)* count_B;
	// Allocate device memory for signal
	Complex* d_A;
	cudaMalloc((void**)&d_A, size_A);
	// Copy host memory to device
	cudaMemcpy(d_A, h_A, size_A,
		cudaMemcpyHostToDevice);

	// Allocate device memory for filter kernel
	Complex* d_B;
	cudaMalloc((void**)&d_B, size_B);

	// Copy host memory to device
	cudaMemcpy(d_B, h_B, size_B,
		cudaMemcpyHostToDevice);

	double *d_C;
	cudaMalloc((void**)&d_C, size_A * sizeof(double));
	double *h_C = (double *)malloc(size_A * sizeof(double));

	// CUFFT plan
	cufftHandle planA, planB;
	cufftPlan1d(&planA, count_A, CUFFT_C2C, 1);
	cufftPlan1d(&planB, count_B, CUFFT_C2C, 1);

	// Transform signal and kernel
	printf("Transforming signal cufftExecC2C\n");
	cufftExecC2C(planA, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_FORWARD);
	cufftExecC2C(planB, (cufftComplex *)d_B, (cufftComplex *)d_B, CUFFT_FORWARD);

	// Multiply the coefficients together and normalize the result
	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");

	ComplexPointwiseMulAndScale << <32, 256 >> >(d_A, d_B, d_C, count_A,count_B);

	// Transform signal back
	printf("Transforming signal back cufftExecC2C\n");
	//cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_INVERSE);

	// Copy device memory to host
	int Min_size = sizeof(Complex) * count_A;
	Complex* h_convolved_signal = (Complex*)malloc(sizeof(Complex) * count_A);
	cudaMemcpy(h_convolved_signal, d_A, size_A,
		cudaMemcpyDeviceToHost);

	for (int i = 0; i < count_A; i++) {
		printf("FFT is %f %f \n", h_convolved_signal[i].x, h_convolved_signal[i].y);
	}

	cudaMemcpy(h_C, d_C, size_A * sizeof(double),cudaMemcpyDeviceToHost);
		printf("LAD IS: %f --- %f --- %f \n", h_C[0], h_C[1],h_C[2]);
	

	// Allocate host memory for the convolution result
	//Complex* h_convolved_signal_ref = (Complex*)malloc(sizeof(Complex) * size_A);

	// Convolve on the host


	//Destroy CUFFT context
	cufftDestroy(planA);
	cufftDestroy(planB);

	// cleanup memory
	free(h_A);
	free(h_B);
	//free(h_padded_signal);
	//free(h_padded_filter_kernel);
	//free(h_convolved_signal_ref);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Clean up memory



	return EXIT_SUCCESS;

}



int readFile(int **grades, char *addr) {
	FILE *fp;
	int temp;
	//grades = NULL;
	int count = 1;
	long index;

	


	fp = fopen(addr, "rb+");

	while (fscanf(fp, "%d", &temp) != EOF)

	{


		if (*grades == NULL)

		{

			*grades = (int *)malloc(sizeof(temp));
			**grades = temp;

			printf("The grade is %d\r\n", temp);
		}

		else
		{
			//printf("The grade is realloc %d\r\n", temp);
			count++;
			*grades = (int *)realloc(*grades, sizeof(int)*count);
			index = count - 1;
			(*grades)[index] = temp;
			//printf("the index is %d\r\n",index);

		}

	}
	printf("Done Total %d numbers \n", count);
	fclose(fp);
	temp = 0;
	/*
	while (index >= 0)
	{

	printf("the read value is %d\r\n", (*grades)[temp]);
	index--;
	temp++;

	} */

	return(count);
	
}


/**
* Program main
*/
int main(int argc, char **argv)
{
	//CompareWav();

	struct dirent *de;  // Pointer for directory entry 

						// opendir() returns a pointer of DIR type.  
	DIR *dr = opendir(".");

	if (dr == NULL)  // opendir returns NULL if couldn't open directory 
	{
		printf("Could not open current directory");
		return 0;
	}


	// for readdir() 
	while ((de = readdir(dr)) != NULL)
		printf("%s\n", de->d_name);

	closedir(dr);

}
