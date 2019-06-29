// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>

// Adds an additional library so that timeGetTime() can be used

#include <stdlib.h>

#include <time.h>
#include <omp.h>

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
#define M_Size 10
__constant__ float M[M_Size];

#define TILE_SIZE 1024
#define MAX_MASK_WIDTH 10


#define blockC 1024
#define gridC 97664
#define nC 4096
#define TILE_WIDTH nC/(blockC*gridC);
__global__ void
convolution_1(int *A, int *B, int *C) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	//float Pvalue = 0;
	if (A[i] = B[i]) {
		C[0] = 5;
	}
}




int readFile(int *grades);






/*

int CompareWav(int *file1, int *file2, int *res)
{

	
	int *h_A;
	int *h_B;

	readFile()

	unsigned int size_A = sizeof(int)* size_A;
	unsigned int size_B = sizeof(int)* size_B;

	// Initialize host memory
	const float valM = 0.01f;
	constantInit(h_N, size_N, 1.0f);
	//constantInit(h_M, size_M, valM);
	float New_M[M_Size];




	// Allocate device memory
	float *d_N, *d_P;

	// Allocate host matrix C
	unsigned int mem_size_P = Width * sizeof(float);
	float *h_P = (float *)malloc(mem_size_P);

	if (h_P == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void **)&d_N, mem_size_N);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}



	error = cudaMalloc((void **)&d_P, mem_size_P);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_N, h_N, mem_size_N, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}



	// Setup execution parameters
	dim3 threads(blockC, 1, 1);
	dim3 grid(gridC, 1, 1);

	fillDataSet(d_N, New_M, Mask_Width, Width);
	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}




	cudaMemcpyToSymbol(M, New_M, sizeof(float)*M_Size);
	// Execute the kernel
	convolution_3 << < grid, threads >> > (d_N, d_P, Mask_Width, Width);
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	printf("Elapsed time in msec = %f\n", msecTotal);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(h_P, d_P, mem_size_P, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}


	// Clean up memory
	free(h_N);
	//free(h_M);
	free(h_P);
	cudaFree(d_N);
	//cudaFree(d_M);
	cudaFree(d_P);


	return EXIT_SUCCESS;

}

*/

int readFile(int **grades) {
	FILE *fp;
	int temp;
	//grades = NULL;
	int count = 1;
	long index;

	


	fp = fopen("CrossingALine.txt", "rb+");

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

	 int *grades=NULL;
	unsigned int size;
	size= readFile(&grades);
	int temp = 0;

	printf("Size is %d \n", &size);

	//temp = 0;
	
	while (size >= 0)
	{
		printf("Size is %d \n", size);
		printf("the read value is %d\r\n", grades[temp]);
		size--;
		temp++;

	}

	free(grades);
	/*

	// By default, we use device 0
	int devID = 0;
	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}
	size_t Mask_Width = M_Size;
	size_t Width = 100000000;
	//float* N=(float *)malloc(sizeof(float) * Width);
	//float* M = (float *)malloc(sizeof(float) * Mask_Width);
	//float* P = (float *)malloc(sizeof(float) * Width);
	// Size of square matrices

	//printf("[-] N = ");
	//scanf("%u", &n);

	

	exit(matrix_result);
	*/
}
