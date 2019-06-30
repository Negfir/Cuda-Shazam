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

static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b)
{
	Complex c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}

static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
	Complex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}

static __global__ void ComplexPointwiseMulAndScale(Complex* A, const Complex* B, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < size; i += numThreads)
		if (i < size) {
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



// Computes convolution on the host
void Convolve(const Complex* signal, int signal_size,
	const Complex* filter_kernel, int filter_kernel_size,
	Complex* filtered_signal)
{
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	// Loop over output element indices
	for (int i = 0; i < signal_size; ++i) {
		filtered_signal[i].x = filtered_signal[i].y = 0;
		// Loop over convolution indices
		for (int j = -maxRadius + 1; j <= minRadius; ++j) {
			int k = i + j;
			if (k >= 0 && k < signal_size)
				filtered_signal[i] = ComplexAdd(filtered_signal[i], ComplexMul(signal[k], filter_kernel[minRadius - j]));
		}
	}
}




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

	// CUFFT plan
	cufftHandle plan;
	cufftPlan1d(&plan, MinCount, CUFFT_C2C, 1);

	// Transform signal and kernel
	printf("Transforming signal cufftExecC2C\n");
	cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_FORWARD);
	cufftExecC2C(plan, (cufftComplex *)d_B, (cufftComplex *)d_B, CUFFT_FORWARD);

	// Multiply the coefficients together and normalize the result
	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
	ComplexPointwiseMulAndScale << <32, 256 >> >(d_A, d_B, MinCount, 1.0f / MinCount);

	// Transform signal back
	printf("Transforming signal back cufftExecC2C\n");
	//cufftExecC2C(plan, (cufftComplex *)d_A, (cufftComplex *)d_A, CUFFT_INVERSE);

	// Copy device memory to host
	int Min_size = sizeof(Complex) * MinCount;
	Complex* h_convolved_signal = (Complex*)malloc(sizeof(Complex) * MinCount);;
	cudaMemcpy(h_convolved_signal, d_A, Min_size,
		cudaMemcpyDeviceToHost);

	for (int i = 0; i < MinCount; i++) {
		printf("FFT is %f %f \n", h_convolved_signal[i].x, h_convolved_signal[i].y);
	}

	// Allocate host memory for the convolution result
	Complex* h_convolved_signal_ref = (Complex*)malloc(sizeof(Complex) * size_A);

	// Convolve on the host
	Convolve(h_A, size_A,
		h_B, size_B,
		h_convolved_signal_ref);

	//Destroy CUFFT context
	cufftDestroy(plan);

	// cleanup memory
	free(h_A);
	free(h_B);
	//free(h_padded_signal);
	//free(h_padded_filter_kernel);
	free(h_convolved_signal_ref);
	cudaFree(d_A);
	cudaFree(d_B);

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
