#include "main.cuh"

__device__ unsigned int dlomutoPartition(unsigned char*& data, unsigned int lo, unsigned int hi) {
    int pivot = data[hi];
    int i = lo;
    int temp = 0;
    for (int j = lo; j < hi; ++j) {
        if (data[j] <= pivot) {
            // Swap with XOR
            // data[i] = data[i] ^ data[j];
            // data[j] = data[i] ^ data[j];
            // data[i] = data[i] ^ data[j];

            // Swap with temp
            temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
        }
    }

    // Swap with XOR
    // data[i] = data[i] ^ data[hi];
    // data[hi] = data[i] ^ data[hi];
    // data[i] = data[i] ^ data[hi];
    
    temp = data[i];
    data[i] = data[hi];
    data[hi] = temp;

    return i;
}

__global__ void CUDAf::dMedianFilter(const unsigned int inputH, const unsigned int inputW, const unsigned int outputH, const unsigned int outputW, const unsigned int kSize, const unsigned int pad, const unsigned int cout, unsigned char* X, unsigned char* Y) {
    unsigned int x_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_global_idx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z_global_idx = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int width = x_global_idx;
    unsigned int height = y_global_idx;
    unsigned int cOut = z_global_idx;

    unsigned int xMapSize = inputH * inputW;
    unsigned int yMapSize = outputH * outputW;

    unsigned char* Temp = new unsigned char[kSize * kSize];

    // As maxpooling store pixel values into a temp array
    unsigned int count = 0;
    unsigned char median_value;
    for (int kh = 0; kh < kSize; ++kh)
    {
        for (int kw = 0; kw < kSize; ++kw)
        {
            int hp = height + kh - pad;
            int wp = width + kw - pad;
            // if (hp >= 0 && wp >= 0 && hp < inputH && wp < inputW) {
            // Should equal to the following statement since casting negative numbers to unsigned will be a vary big number
            if ((unsigned int)hp < inputH && (unsigned int)wp < inputW) {
                Temp[count++] = X[cOut * xMapSize + hp * inputW + wp];
                //count++;
            }
        }
    }

    // Quick Select
    unsigned int key = (count) / 2; // median index of (count + 1) / 2 therefore +1 is not needed 
    unsigned int l = 0; // index
    unsigned int r = count - 1; // index

    if (l == r) {
        median_value = Temp[key];
    }
    else {
        while (l < r) {
            unsigned int pivotIdx = dlomutoPartition(Temp, l, r);

            if (key == pivotIdx) {
                median_value = Temp[key];
                break;
            }
            else if (key < pivotIdx) {
                r = pivotIdx - 1;
            }
            else {
                l = pivotIdx + 1;
            }
        }
    }

    Y[cOut * yMapSize + height * outputW + width] = median_value;

    delete[] Temp;
}

__global__ void CUDAf::dMeanFilter(const unsigned int inputH, const unsigned int inputW, const unsigned int outputH, const unsigned int outputW, const unsigned int kSize, const unsigned int pad, const unsigned int cout, unsigned char* X, unsigned char* Y) {
    unsigned int x_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y_global_idx = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int z_global_idx = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int width = x_global_idx;
    unsigned int height = y_global_idx;
    unsigned int cOut = z_global_idx;

    unsigned int xMapSize = inputH * inputW;
    unsigned int yMapSize = outputH * outputW;

    unsigned int sum = 0;
    for (int kh = 0; kh < kSize; ++kh)
    {
        for (int kw = 0; kw < kSize; ++kw)
        {
            int hp = height + kh - pad;
            int wp = width + kw - pad;
            // if (hp >= 0 && wp >= 0 && hp < inputH && wp < inputW) {
            // Should equal to the following statement since casting negative numbers to unsigned will be a vary big number
            if ((unsigned int)hp < inputH && (unsigned int)wp < inputW) {
                sum += X[cOut * xMapSize + hp * inputW + wp];
            }
        }
    }

    Y[cOut * yMapSize + height * outputW + width] = sum / (kSize * kSize);
}

cv::Mat cvf::cvReadImg(const char* filename, unsigned char*& buffer) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);

    if (buffer == nullptr) {
        printf("image size : %d\n", image.rows * image.cols * image.channels());
        buffer = new unsigned char[image.rows * image.cols * image.channels()];
    }

    #pragma omp simd parallel
    for (int c = 0; c < image.channels(); ++c) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                buffer[c * image.rows * image.cols + y * image.cols + x] = image.at<cv::Vec3b>(y, x)[c];
            }
        }
    }

    return image.clone();
}

void cvf::cvOutputImg(const char* filename, cv::Mat& image, unsigned char*& buffer) {
    std::cout << image.rows << " " << image.cols << " " << image.channels() << "\n\n";

    if (buffer == nullptr) {
        std::cout << "Error\n\n";
    }

    #pragma omp simd parallel
    for (int c = 0; c < image.channels(); ++c) {
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                image.at<cv::Vec3b>(y, x)[c] = buffer[c * image.rows * image.cols + y * image.cols + x];
            }
        }
    }

    delete[] buffer;
    buffer = nullptr;

    cv::imwrite(filename, image);
}

void CUDAf::getDeviceName()
{
	printf("CUDA Device Info\n");

	int deviceCount = 0;
	cudaError_t error_id;

	error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("Error cudaGetDeviceCount() : %d\n%s\n\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0) {
		printf("There are no available device(s) that support CUDA\n");
		exit(EXIT_FAILURE);
	}
	else 
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);

	cudaDeviceProp deviceProp;

	// Iterate through all the devices found
	for (int i = 0; i < deviceCount; ++i) {
		cudaSetDevice(i);
		cudaGetDeviceProperties(&deviceProp, i);
		printf("Device: %d, %s\n\n", i, deviceProp.name);
	}
}
