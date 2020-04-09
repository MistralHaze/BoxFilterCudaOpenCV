//****************************************************************************
// Also note that we've supplied a helpful debugging function called
// checkCudaErrors. You should wrap your allocation and copying statements like
// we've done in the code we're supplying you. Here is an example of the unsafe
// way to allocate memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows *
// numCols));
//****************************************************************************

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iomanip>
#include <iostream>

#define BLOCK_WIDTH 64
#define BLOCK_HEIGHT 16

//__constant__ float d_filter[KernelWidth * KernelWidth];

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void makeLaplacianFilter(float** h_filter);
void makeHorizontalLineFilter(float** h_filter);
void makeSharpnessFilter(float** h_filter, int type);
void makeBlurFilter(float** h_filter, int type);

template <typename T>
void check(T err, const char* const func, const char* const file,
	const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

// clamp x to range [a, b]
__device__ float clamp(float x, float a, float b)
{
	return fmax(a, fmin(b, x));
}

__global__ void box_filter(const unsigned char* const inputChannel,
	unsigned char* const outputChannel, int numRows,
	int numCols, const float* const filter,
	const int filterWidth) {
	// TODO:
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la
	// imagen
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	// make sure we don't try and access memory outside the image
	// by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	int kernelWidth = (filterWidth - 1) / 2;
	unsigned char convolutionSum = 0;
	for (int i = -kernelWidth; i <= kernelWidth; ++i)
	{
		for (int j = -kernelWidth; j <= kernelWidth; ++j)
		{
			if ((thread_2D_pos.x + i) >= numCols || (thread_2D_pos.x + i) < 0 || (thread_2D_pos.y + j) >= numRows || (thread_2D_pos.y + j) < 0)
				convolutionSum += 0;
			else
				convolutionSum += inputChannel[(thread_2D_pos.y + j) * numCols + (thread_2D_pos.x + i)] * filter[(i + kernelWidth) * filterWidth + (j + kernelWidth)];
		}
	}

	convolutionSum = clamp(convolutionSum, 0, 255);

	outputChannel[thread_1D_pos] = convolutionSum;

	// NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que
	// al aplicar el filtro los valores de sus vecinos sean correctos, ya que
	// pueden salirse de la imagen.
}

// This kernel takes in an image represented as a uchar4 and splits
// it into three images consisting of only one color channel each
__global__ void separateChannels(const uchar4* const inputImageRGBA,
	int numRows, int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel) {
	// TODO:
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la
	// imagen
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	// make sure we don't try and access memory outside the image
	// by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
	greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
	blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

// This kernel takes in three color channels and recombines them
// into one image. The alpha channel is set to 255 to represent
// that this image has no transparency.
__global__ void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA, int numRows,
	int numCols) {
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	// make sure we don't try and access memory outside the image
	// by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows) return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	// Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char* d_red, * d_green, * d_blue;
float* d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage,
	const size_t numColsImage,
	const float* const h_filter,
	const size_t filterWidth) {
	// allocate memory for the three different channels
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

	// TODO:
	// Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada
	// Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)

	checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

	//cudaMemcpyToSymbol(d_filter, &h_filter, sizeof(float) * filterWidth * filterWidth);
}

const int KernelWidth = 3;  // OJO CON EL TAMAÑO DEL FILTRO//

void create_filter(float** h_filter, int* filterWidth) {
	// create and fill the filter we will convolve with
	*h_filter = new float[KernelWidth * KernelWidth];

	//5*5
	//makeLaplacianFilter(h_filter);

	//3*3
	makeSharpnessFilter(h_filter, 0);
	//makeHorizontalLineFilter( h_filter);
	//Filtro gaussiano: blur
	/*
	const float KernelSigma = 2.;

	float filterSum = 0.f; //for normalization

	for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r)
	{
		for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c)
		{
			float filterValue = expf(-(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
			(*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] = filterValue; filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -KernelWidth / 2; r <= KernelWidth / 2; ++r)
	{
		for (int c = -KernelWidth / 2; c <= KernelWidth / 2; ++c)
		{
			(*h_filter)[(r + KernelWidth / 2) * KernelWidth + c + KernelWidth / 2] *= normalizationFactor;
		}
	}
	*/

	// TODO: crear los filtros segun necesidad
	// NOTA: cuidado al establecer el tamaño del filtro a utilizar

	* filterWidth = KernelWidth;
}

void convolution(const uchar4* const h_inputImageRGBA,
	uchar4* const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA, const size_t numRows,
	const size_t numCols, unsigned char* d_redFiltered,
	unsigned char* d_greenFiltered, unsigned char* d_blueFiltered,
	const int filterWidth) {
	// TODO: Calcular tamaños de bloque
	const dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	const dim3 gridSize(ceil((float)numCols / BLOCK_WIDTH), ceil((float)numRows / BLOCK_HEIGHT), 1);

	// TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
	separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

	// TODO: Ejecutar convolución. Una por canal
	box_filter << <gridSize, blockSize >> > (d_red, d_redFiltered, numRows, numCols, d_filter, filterWidth);
	box_filter << <gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
	box_filter << <gridSize, blockSize >> > (d_blue, d_blueFiltered, numRows, numCols, d_filter, filterWidth);

	// Recombining the results.
	recombineChannels << <gridSize, blockSize >> > (d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA, numRows, numCols);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

// Free all the memory that we allocated
// TODO: make sure you free any arrays that you allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
}

void makeLaplacianFilter(float** h_filter)
{
	// Laplaciano 5x5
	(*h_filter)[0] = 0;
	(*h_filter)[1] = 0;
	(*h_filter)[2] = -1.;
	(*h_filter)[3] = 0;
	(*h_filter)[4] = 0;
	(*h_filter)[5] = 1.;
	(*h_filter)[6] = -1.;
	(*h_filter)[7] = -2.;
	(*h_filter)[8] = -1.;
	(*h_filter)[9] = 0;
	(*h_filter)[10] = -1.;
	(*h_filter)[11] = -2.;
	(*h_filter)[12] = 17.;
	(*h_filter)[13] = -2.;
	(*h_filter)[14] = -1.;
	(*h_filter)[15] = 1.;
	(*h_filter)[16] = -1.;
	(*h_filter)[17] = -2.;
	(*h_filter)[18] = -1.;
	(*h_filter)[19] = 0;
	(*h_filter)[20] = 1.;
	(*h_filter)[21] = 0;
	(*h_filter)[22] = -1.;
	(*h_filter)[23] = 0;
	(*h_filter)[24] = 0;
}

void makeHorizontalLineFilter(float** h_filter)
{
	//Detección linea horizontal
	(*h_filter)[0] = -1;
	(*h_filter)[1] = -1;
	(*h_filter)[2] = -1;
	(*h_filter)[3] = 2;
	(*h_filter)[4] = 2;
	(*h_filter)[5] = 2.;
	(*h_filter)[6] = -1.;
	(*h_filter)[7] = -1.;
	(*h_filter)[8] = -1.;
}

void makeSharpnessFilter(float** h_filter, int type)
{
	//Filtro de nitidez de paso alto 3*3
	switch (type)
	{
	default:
	case 0://Aumentar nitidez
		(*h_filter)[0] = 0;
		(*h_filter)[1] = -0.25;
		(*h_filter)[2] = 0;
		(*h_filter)[3] = -0.25;
		(*h_filter)[4] = 2;
		(*h_filter)[5] = -0.25;
		(*h_filter)[6] = 0;
		(*h_filter)[7] = -0.25;
		(*h_filter)[8] = 0;
		break;
	case 1://Aumentar nitidez II
		(*h_filter)[0] = -0.25;
		(*h_filter)[1] = -0.25;
		(*h_filter)[2] = -0.25;
		(*h_filter)[3] = -0.25;
		(*h_filter)[4] = 3;
		(*h_filter)[5] = -0.25;
		(*h_filter)[6] = -0.25;
		(*h_filter)[7] = -0.25;
		(*h_filter)[8] = -0.25;
		break;
	case 2: //Nitidez 3*3 Filtro paso alto
		(*h_filter)[0] = -1;
		(*h_filter)[1] = -1;
		(*h_filter)[2] = -1;
		(*h_filter)[3] = -1;
		(*h_filter)[4] = 9;
		(*h_filter)[5] = -1;
		(*h_filter)[6] = -1;
		(*h_filter)[7] = -1;
		(*h_filter)[8] = -1;
		break;
	}
}

void makeBlurFilter(float** h_filter, int type)
{
	//filtro de suavizado
	switch (type)
	{
	default:
	case 0://Media aritmetica suave
		(*h_filter)[0] = 0.111;
		(*h_filter)[1] = 0.111;
		(*h_filter)[2] = 0.111;
		(*h_filter)[3] = 0.111;
		(*h_filter)[4] = 0.111;
		(*h_filter)[5] = 0.111;
		(*h_filter)[6] = 0.111;
		(*h_filter)[7] = 0.111;
		(*h_filter)[8] = 0.111;
		break;
	case 1://Suavizado 3*3
		(*h_filter)[0] = 1;
		(*h_filter)[1] = 2;
		(*h_filter)[2] = 1;
		(*h_filter)[3] = 2;
		(*h_filter)[4] = 4;
		(*h_filter)[5] = 2;
		(*h_filter)[6] = 1;
		(*h_filter)[7] = 2;
		(*h_filter)[8] = 1;
		break;
	}
}