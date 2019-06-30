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

// Adds an additional library so that timeGetTime() can be used

#include <stdlib.h>

#include <time.h>
#include <omp.h>

#define SIGNAL_SIZE        50
#define FILTER_KERNEL_SIZE 11

typedef float2 Complex;
/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/

#define NX 256
#define BATCH 1

__global__ void
compareKernel(Complex *A, Complex *B, int Size) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	//int Pvalue = 0;
	if (i < Size) {
		fabs(A[i].x - B[i].x);
	}
		
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
		h_A[i].x = h_A_real[i] + 0.0 ;
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
	Complex* h_padded_A;
	Complex* h_padded_B;
	int new_size = PadData(h_signal, &h_padded_signal, SIGNAL_SIZE,
		h_filter_kernel, &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
	int mem_size = sizeof(Complex) * new_size;
	*/

	unsigned int MinCount= count_B;

	if (count_A < count_B) {
		MinCount = count_A;
	}

	unsigned int MaxCount = count_B;

	if (count_A > count_B) {
		MaxCount = count_A;
	}



	// Allocate device memory
	double *d_C;
	Complex *d_A, *d_B;

	// Allocate host matrix C
	unsigned int size_C = sizeof(double)* MinCount;
	double *h_C = (double *)malloc(size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	unsigned int size_A = sizeof(Complex)* count_A;
	unsigned int size_B = sizeof(Complex)* count_B;

	// Pad signal and filter kernel
	Complex* h_padded_A;
	Complex* h_padded_B;
	int new_size = PadData(h_A, &h_padded_A, size_A,
		h_B, &h_padded_B, size_A);
	int mem_size = sizeof(Complex) * new_size;

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

	//----------------------------------------------------------------------------
	printf("[simpleCUFFT] is starting...\n");
	


	// CUFFT plan
	cufftHandle plan;
	cufftPlan1d(&plan, MinCount, CUFFT_C2C, 1);

	// Transform signal and kernel
	printf("Transforming signal cufftExecC2C\n");
	cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_FORWARD);
	cufftExecC2C(plan, (cufftComplex *)d_B, (cufftComplex *)d_B, CUFFT_FORWARD);

	//printf("Transforming signal back cufftExecC2C\n");
	//cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_INVERSE);
	//Complex* tmp = (Complex*)malloc(sizeof(Complex) * new_size);

	/*
	int Max_size = sizeof(Complex) * MaxCount;
	Complex *h_convolved_signal = (Complex*)malloc(sizeof(Complex) * MaxCount);
	cudaMemcpy(h_convolved_signal, d_A, mem_size,
		cudaMemcpyDeviceToHost);
	

	for (int i = 0; i < count_A; i++) {
		printf("FFT is %f %f \n", h_convolved_signal[i].x, h_convolved_signal[i].y);
	}
	
	*/
	
	/*
	cufftHandle plan;
	cufftPlan1d(&plan, new_size, CUFFT_C2C, 1);

	cufftComplex* outRes;

	// Transform signal and kernel
	printf("Transforming signal cufftExecC2C\n");
	cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_FORWARD);
	cufftExecC2C(plan, (cufftComplex *)d_B, (cufftComplex *)d_B, CUFFT_FORWARD);

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		exit(EXIT_FAILURE);
	}

	printf("FFT Done \n");

	printf("Transforming signal back cufftExecC2C\n");
	cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_INVERSE);

	Complex* TestA = h_padded_A;
	error=cudaMemcpy(TestA, d_A, mem_size,
		cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}


	for (int i = 1; i < 3; i++) {
		printf("FFT is %d %d \n", TestA[i].x, TestA[i].y);
	}
	
	/*
	// Multiply the coefficients together and normalize the result
	
	//ComplexPointwiseMulAndScale << <32, 256 >> >(d_signal, d_filter_kernel, new_size, 1.0f / new_size);

	// Transform signal back
	//printf("Transforming signal back cufftExecC2C\n");
	//cufftExecC2C(plan, (cufftComplex *)d_signal, (cufftComplex *)d_signal, CUFFT_INVERSE);
	//----------------------------------------------------------------------------

	// -------------cuFFT IS HERE ------------------
	/*
	cufftHandle plan;
	cufftComplex *data;

	//cufftHandle plan;
	//cufftPlan1d(&plan, new_size, CUFFT_C2C, 1);

	if (cufftPlan1d(&plan, size_A, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: Plan creation failed");
		exit(EXIT_FAILURE);
	}
	
	//cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_FORWARD);
	if (cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
		exit(EXIT_FAILURE);
	}
	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		exit(EXIT_FAILURE);
	}

	
	printf("cuFFT Done :)");
	//cufftComplex d= (cufftComplex*)d_A;

	/*
	for (int i = 0; i < count_A; i++) {
		printf("FFT is %d %d \n", d_A->x, d_A->y);
	}

	*/

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

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to synchronize\n");
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	compareKernel << < grid, threads >> > (d_A, d_B, MinCount);
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
