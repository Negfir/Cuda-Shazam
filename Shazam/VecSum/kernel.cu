// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cufft.h>
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

#define NX 256
#define BATCH 1

__global__ void
compareKernel(int *A, int *B, int *C) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	//int Pvalue = 0;
	C[i] = abs(A[i] - B[i]);

		
}




int readFile(int **grades, char *addr);








int CompareWav()
{
	
	int *h_A = NULL;
	unsigned int count_A;
	count_A = readFile(&h_A, "M1.txt");
	
	
	int *h_B = NULL;
	unsigned int count_B;
	count_B = readFile(&h_B, "M2.txt");


	unsigned int size_A = sizeof(int)* count_A;
	unsigned int size_B = sizeof(int)* count_B;

	unsigned int MinCount= count_B;

	if (count_A < count_B) {
		MinCount = count_A;
	}

	unsigned int MaxCount = count_B;

	if (count_A > count_B) {
		MaxCount = count_A;
	}



	// Allocate device memory
	int *d_A, *d_B, *d_C;

	// Allocate host matrix C
	unsigned int size_C = sizeof(int)* MinCount;
	int *h_C = (int *)malloc(size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void **)&d_A, size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// -------------cuFFT IS HERE ------------------
	
	cufftHandle plan;
	cufftComplex *data;
	/*
	cudaMalloc((void**)&data, sizeof(cufftComplex)*(NX / 2 + 1)*BATCH);
	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;
	}
	*/
	if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		exit(EXIT_FAILURE);
	}
	
		
	if (cufftExecR2C(plan, (cufftReal*)d_A, (cufftComplex*)d_A) != CUFFT_SUCCESS) {
			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
			exit(EXIT_FAILURE);
		}
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		exit(EXIT_FAILURE);
	}
	
	printf("cuFFT Done :)");

	int gridCount = ceil(MinCount / 1024);
	// Setup execution parameters
	dim3 threads(1024, 1, 1);
	dim3 grid(gridCount, 1, 1);

	
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


	// Execute the kernel
	//compareKernel << < grid, threads >> > (d_A, d_B, d_C);
	//cudaDeviceSynchronize();
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
	error = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}


	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);


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
	CompareWav();
	/*
	char *addr = argv[1];
	 int *grades=NULL;
	 unsigned int size;
	size= readFile(&grades,"M2.txt");
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
	*/


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
