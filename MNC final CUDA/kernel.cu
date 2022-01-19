
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

cudaError_t matrixMultiplication(const float *a, const float *b, float *c, unsigned int N, unsigned int M, unsigned int P);
cudaError_t transpose(const float* a, float* at, unsigned int N, unsigned int M);
cudaError_t matrixDivision(const float* a, const float m, float* result, unsigned int N, unsigned int M);
cudaError_t centerMatrix(const float* matrix, float* centeredMatrix, unsigned int N, unsigned int M);
int fileReadTest();
int matrixOpsTest();
int pca();

__global__ void matMultKernel(float* c, const float* a, const float* b, const int N, const int M, const int P) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (i < N && j < P) {
		float sum = 0;
		for (int k = 0; k < M; k++) {
			sum += a[i*M + k] * b[k*N + j];
		}
		c[i*N + j] = sum;
	}
}

__global__ void transposeKernel(const float* a, float* at, const int N, const int M) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < M) at[i + j * N] = a[j + i * M];
}

__global__ void matrixDivisionKernel(const float* a, const float m, float* result, const int N, const int M) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < M) result[i + j * N] = a[i + j * N] / m;
}

__global__ void centerMatrixKernel(const float* matrix, float* centeredMatrix, const int N, const int M) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int column = blockIdx.y * blockDim.y + threadIdx.y;

	if (column < M && row < N) {
		float mean = 0;
		for (int i = 0; i < N; i++) {
			mean += matrix[column + i * M];
		}
		mean /= N;

		centeredMatrix[column + row * M] = matrix[column + row * M] - mean;
	}
}

float* fileToMatrix(char* filename, char* format, int lines, int fields) {
	FILE* file;
	file = fopen(filename, "r");
	if (file == NULL) {
		printf("Error: could not open file %s\n", filename);
		return NULL;
	}

	float* matrix = (float*)malloc(sizeof(float) * fields * lines);
	for (int line = 0; line < lines; line++) {
		for (int field = 0; field < fields; field++) {
			fscanf(file, format, &matrix[line * fields + field]);
		}
		fscanf(file, "%*d");
	}

	return matrix;
}

void showMatrix(float* matrix, int width, int height) {
	int p = 0;
	for (int i = 0; i < width * height; i++) {
		printf("%f ", matrix[i]);
		p++;
		if (p >= height) {
			printf("\n");
			p = 0;
		}
	}
	printf("\n");
}

void showArray(float* array, int arraySize) {
	for (int i = 0; i < arraySize; i++) {
		printf("%f ", array[i]);
	}
	printf("\n\n");
}

float* centerMatrixSequential(float* matrix, const int N, const int M) {
	float* means = (float*)malloc(sizeof(float) * M);
	for (int column = 0; column < M; column++) {
		means[column] = 0;
		for (int row = 0; row < N; row++) {
			means[column] += matrix[column + row * M];
		}
		means[column] /= N;
	}
	showArray(means, M);

	float* centeredMatrix = (float*)malloc(sizeof(float) * N * M);
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			centeredMatrix[i + j * M] = matrix[i + j * M] - means[i];
		}
	}
	free(means);
	return centeredMatrix;
}

int main()
{
	pca();
}

int pca() {
	int attributes = 7;
	int instances = 210;
	float* seeds = fileToMatrix("seeds_dataset.txt", "%f", instances, attributes);

	float* seedsCentered = (float*)malloc(sizeof(float) * instances * attributes);
	centerMatrix(seeds, seedsCentered, instances, attributes);

	float* seedsT = (float*)malloc(sizeof(float) * instances * attributes);
	float* ZUnscaled = (float*)malloc(sizeof(float) * attributes * attributes);
	float* Z = (float*)malloc(sizeof(float) * attributes * attributes);

	transpose(seedsCentered, seedsT, instances, attributes);
	matrixMultiplication(seedsT, seedsCentered, ZUnscaled, attributes, instances, attributes);
	matrixDivision(ZUnscaled, instances, Z, attributes, attributes);

	printf("Matriz de covarianza:\n");
	showMatrix(Z, attributes, attributes);

	free(seeds);
	free(seedsCentered);
	free(seedsT);
	free(ZUnscaled);
	free(Z);
	return 0;
}

int fileReadTest() {
	int instances = 210;
	int attributes = 7;
	float* seeds = fileToMatrix("seeds_dataset.txt", "%f", instances, attributes);
	showMatrix(seeds, instances, attributes);
	free(seeds);
	return 0;
}

int matrixOpsTest() {
	const int N = 7;
	const int M = 210;
	float* a = (float*)malloc(sizeof(float) * N * M);
	float* b = (float*)malloc(sizeof(float) * M * N);
	float* c = (float*)malloc(sizeof(float) * N * N);
	float* at = (float*)malloc(sizeof(float) * N * M);
	float* div = (float*)malloc(sizeof(float) * N * M);

	for (int y = 0; y < M; y++) {
		for (int x = 0; x < N; x++) {
			a[x + y * N] = x + y * N;
			b[x + y * N] = x + y * N;
		}
	}

	transpose(a, at, N, M);
	matrixMultiplication(a, at, c, N, M, N);
	matrixDivision(c, M, div, N, N);

	showMatrix(a, N, M);
	showMatrix(b, M, N);
	showMatrix(c, N, N);
	showMatrix(at, N, M);
	showMatrix(div, N, N);

	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

cudaError_t matrixMultiplication(const float* a, const float* b, float* c, unsigned int N, unsigned int M, unsigned int P) {
	float* dev_a = 0;
	float* dev_b = 0;
	float* dev_c = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_c, N * P * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_a, N * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_b, M * P * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, N * M * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, M * P * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int numBlocksX = (int)ceil(N / 32.0);
	int numBlocksY = (int)ceil(P / 32.0);
	dim3 blocks(numBlocksX, numBlocksY);
	dim3 threadsPerBlock(32, 32);
	matMultKernel << <blocks, threadsPerBlock >> > (dev_c, dev_a, dev_b, N, M, P);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matrixMultiplicator launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matrixMultiplicator!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(c, dev_c, N * P * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

cudaError_t transpose(const float* a, float* at, unsigned int N, unsigned int M) {
	float* dev_a = 0;
	float* dev_at = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_a, N * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_at, N * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, N * M * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int numBlocksX = (int)ceil(N / 32.0);
	int numBlocksY = (int)ceil(M / 32.0);
	dim3 blocks(numBlocksX, numBlocksY);
	dim3 threadsPerBlock(32, 32);
	transposeKernel << <blocks, threadsPerBlock >> > (dev_a, dev_at, N, M);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matrixMultiplicator launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matrixMultiplicator!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(at, dev_at, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_at);

	return cudaStatus;
}

cudaError_t matrixDivision(const float* a, const float m, float* result, unsigned int N, unsigned int M)
{
	float* dev_a = 0;
	float* dev_result = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_a, N * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_result, N * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_a, a, N * M * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int numBlocksX = (int)ceil(N / 32.0);
	int numBlocksY = (int)ceil(M / 32.0);
	dim3 blocks(numBlocksX, numBlocksY);
	dim3 threadsPerBlock(32, 32);
	matrixDivisionKernel << <blocks, threadsPerBlock >> > (dev_a, m, dev_result, N, M);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matrixMultiplicator launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matrixMultiplicator!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(result, dev_result, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_a);
	cudaFree(dev_result);

	return cudaStatus;
}

cudaError_t centerMatrix(const float* matrix, float* centeredMatrix, unsigned int N, unsigned int M)
{
	float* dev_matrix = 0;
	float* dev_centered_matrix = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_matrix, N * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)& dev_centered_matrix, N * M * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_matrix, matrix, N * M * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int numBlocksX = (int)ceil(N / 32.0);
	int numBlocksY = (int)ceil(M / 32.0);
	dim3 blocks(numBlocksX, numBlocksY);
	dim3 threadsPerBlock(32, 32);
	centerMatrixKernel << <blocks, threadsPerBlock >> > (dev_matrix, dev_centered_matrix, N, M);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "matrixMultiplicator launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching matrixMultiplicator!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(centeredMatrix, dev_centered_matrix, N * M * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_matrix);
	cudaFree(dev_centered_matrix);

	return cudaStatus;
}

