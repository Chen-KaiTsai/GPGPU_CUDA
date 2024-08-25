#include "main.cuh"

int main(int argc, char** argv)
{
    CUDAf::getDeviceName();
    printf("Use cuda device 0\n");

    unsigned char* buffer = nullptr;

    printf("Reading Image\n");
    cv::Mat image = cvf::cvReadImg("Enhanced_CPP_output.png", buffer);

    printf("Image info :\nWidth = %d\nHeight = %d\nChannel = %d\n", image.cols, image.rows, image.channels());

    cudaError_t error;

    unsigned char* dInput = nullptr;
    unsigned char* dOutput = nullptr;
    size_t buffer_size = image.rows * image.cols * image.channels();
    
    // Transfer Mem to GPU
    error = cudaMalloc(&dInput, buffer_size);
    if (error != cudaSuccess) {
        printf("Error dInput cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMemcpy(dInput, buffer, buffer_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dInput cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMalloc(&dOutput, buffer_size);
    if (error != cudaSuccess) {
        printf("Error dOutput cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    // Invoke Kernel
    dim3 dimBlock = {16, 8, 3};
    dim3 dimGrid;
    dimGrid.x = (image.cols + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (image.rows + dimBlock.y - 1) / dimBlock.y;
    dimGrid.z = (image.channels() + dimBlock.z - 1) / dimBlock.z;

    //printf("dimGrid [%d, %d, %d]\n", dimGrid.x, dimGrid.y, dimGrid.z);
    
    CUDAf::dMeanFilter<<<dimGrid, dimBlock>>>(image.rows, image.cols, image.rows, image.cols, 5, 2, image.channels(), dInput, dOutput);
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error Kernel : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    // Transfer Back to CPU
    error = cudaMemcpy(buffer, dOutput, buffer_size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error buffer cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    printf("Output Image\n");
    cvf::cvOutputImg("dMeanFilterOutput.png", image, buffer);

    if (buffer != nullptr) {
        delete[] buffer;
    }

    if (dInput != nullptr) {
        error = cudaFree(dInput);
        if (error != cudaSuccess) {
            printf("Error dInput cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }

    if (dOutput != nullptr) {
        error = cudaFree(dOutput);
        if (error != cudaSuccess) {
            printf("Error dOutput cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }

    return 0;
}
