
#include "CuProbabilityTree.h"

#ifdef ENABLE_CUDA

#include <iostream>
#include <iterator> // for std::distance
#include <vector>
#include <curand_kernel.h>

namespace galsim
{

    // CUDA 内核函数，用于生成均匀分布在单位圆内的点
    __global__ void radial_rand_shoot_kernel(long seed, 
        double * x, double* y, double* flux, int N, double fluxPerPhoton,  // output
        DeviceElement** d_shortcut, int shortcutSize, double totalAbsFlux  // find
        ) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            curandState state;
            curand_init(seed, idx, 0, &state);

            double xu, yu, sq;
            do {
                xu = 2.0 * curand_uniform(&state) - 1.0;
                yu = 2.0 * curand_uniform(&state) - 1.0;
                sq = xu * xu + yu * yu;
            } while (sq >= 1.0 || sq == 0.0);

            double unitRandom = sq;

            // find 
            int i = int(unitRandom * shortcutSize);
            DeviceElement* element = d_shortcut[i];
            unitRandom *= totalAbsFlux;

            // 用栈来模拟递归
            while (element->left || element->right) {
                if (unitRandom < element->right->leftAbsFlux) {
                    element = element->left;
                } else {
                    element = element->right;
                }
            }
            unitRandom = (unitRandom - element->leftAbsFlux) / element->absFlux;
            Device_Interval * data =  element->data;

            // interpolateFlux
            double fraction = unitRandom;
            double radius, flux_edge;
            if (data->_isRadial) {
                double d = data->_d * fraction;
                double dr = 2.0 * d / (sqrt(4.0 * data->_b * d + data->_c * data->_c) + data->_c);
                double delta = 0.;
                do {
                    double df = dr * (data->_c + dr * (data->_b + data->_a * dr)) - d;
                    double dfddr = data->_c + dr * (2.0 * data->_b + 3.0 * data->_a * dr);
                    delta = df / dfddr;
                    dr -= delta;
                } while (fabs(delta) > data->shoot_accuracy);
                radius = data->_xLower + data->_xRange * dr;
            } else {
                double c = fraction * data->_c;
                double dx = c / (sqrt(data->_a * c + data->_b * data->_b) + data->_b);
                radius = data->_xLower + data->_xRange * dx;
            }
            flux_edge =  data->_flux < 0 ? -1. : 1.;
            // rScale
            double rScale = radius / std::sqrt(sq);
            
            x[idx] = xu*rScale;
            y[idx] = yu*rScale;
            flux[idx] = flux_edge*fluxPerPhoton; 
        }
    }


 // CUDA 内核函数，用于生成均匀分布在单位圆内的点
    __global__ void xandy_rand_shoot_kernel(long seed, bool xandy,
        double * x, double* y, double* flux, int N, double fluxPerPhoton,  // output
        DeviceElement** d_shortcut, int shortcutSize, double totalAbsFlux  // find
        ) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            curandState state;
            curand_init(seed, idx, 0, &state);

            double unitRandom = curand_uniform(&state) ;

            // find 
            int i = int(unitRandom * shortcutSize);
            DeviceElement* element = d_shortcut[i];
            unitRandom *= totalAbsFlux;

            // 用栈来模拟递归
            while (element->left || element->right) {
                if (unitRandom < element->right->leftAbsFlux) {
                    element = element->left;
                } else {
                    element = element->right;
                }
            }
            unitRandom = (unitRandom - element->leftAbsFlux) / element->absFlux;
            Device_Interval * data =  element->data;

            // interpolateFlux
            double c = unitRandom * data->_c;
            double dx = c / (sqrt(data->_a * c + data->_b * data->_b) + data->_b);

            double xi = data->_xLower + data->_xRange * dx;
            double flux_xi =  data->_flux < 0 ? -1. : 1.;

            double yi = 0.;
            double flux_yi = 1.0;
            if (xandy) { 
                unitRandom = curand_uniform(&state) ;
                // find 
                int i = int(unitRandom * shortcutSize);
                DeviceElement* element = d_shortcut[i];
                unitRandom *= totalAbsFlux;

                // 用栈来模拟递归
                while (element->left || element->right) {
                    if (unitRandom < element->right->leftAbsFlux) {
                        element = element->left;
                    } else {
                        element = element->right;
                    }
                }
                unitRandom = (unitRandom - element->leftAbsFlux) / element->absFlux;
                Device_Interval * data =  element->data;


                c = unitRandom * data->_c;
                dx = c / (sqrt(data->_a * c + data->_b * data->_b) + data->_b);
                yi = data->_xLower + data->_xRange * dx; 
                flux_yi =  data->_flux < 0 ? -1. : 1.;
            }

            x[idx] = xi;
            y[idx] = yi;
            flux[idx] = flux_xi* flux_yi*fluxPerPhoton; 
        }
    }


    void CuIntervalProbabilityTree::find_and_interpolateFlux(long seed, double * x, double* y, double* flux, int N, 
                    double fluxPerPhoton, const bool isRadial, bool xandy) const
    {
        time_t start, end;
        start = clock();


        int blockSize = 256; // Example block size
        int numBlocks = (N + blockSize - 1) / blockSize;
        if(isRadial)
        {
            radial_rand_shoot_kernel<<<numBlocks, blockSize>>>(seed, x, y, flux, N, fluxPerPhoton, 
                        _d_shortcut, _shortcutSize, this->_totalAbsFlux);
        }
        else
        {
            xandy_rand_shoot_kernel<<<numBlocks, blockSize>>>(seed, xandy, x, y, flux, N, fluxPerPhoton, 
                        _d_shortcut, _shortcutSize, this->_totalAbsFlux);
        }
        CUDA_CHECK_RETURN(cudaGetLastError());        

        end = clock();
        double time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        // printf("find_and_interpolateFlux time: %f ms,    %d\n", time, N);
    }



    void CuIntervalProbabilityTree::copyNodesToGPU(const Element* cpuElement, 
            DeviceElement * & d_elements_iter, Device_Interval * & d_interval_iter,  DeviceElement*& currentGPUElement) 
    {
        double absflux =  cpuElement->getAbsFlux();
        double leftabsflux = cpuElement->getLeftAbsFlux();
        CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->leftAbsFlux, &leftabsflux, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->absFlux, &absflux, sizeof(double), cudaMemcpyHostToDevice));

        Interval * ptr = cpuElement->getData().get(); 
        // printf("%p\n", ptr);
        if(ptr != nullptr) {
            Device_Interval host;
            ptr->get_interval_data(host);
            Device_Interval * dev = d_interval_iter++;
            CUDA_CHECK_RETURN(cudaMemcpy(dev, &host, sizeof(Device_Interval), cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->data, &dev, sizeof(Device_Interval*), cudaMemcpyHostToDevice));
        }

        if (cpuElement->isLeaf()) {

            DeviceElement* next = nullptr;
            CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->left, &next, sizeof(DeviceElement*), cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->right, &next, sizeof(DeviceElement*), cudaMemcpyHostToDevice));       
            // 拷贝其他必要的数据...

        } else {
            // 设置指针并递归拷贝左右子树

            // 设置指针
            DeviceElement* next = d_elements_iter;
            CUDA_CHECK_RETURN(cudaMemcpy(&(currentGPUElement->left), &next, sizeof(DeviceElement*), cudaMemcpyHostToDevice));

            d_elements_iter++;
            copyNodesToGPU(cpuElement->getLeft(), d_elements_iter, d_interval_iter, next);

            next = d_elements_iter;
            CUDA_CHECK_RETURN(cudaMemcpy(&(currentGPUElement->right), &next, sizeof(DeviceElement*), cudaMemcpyHostToDevice));

            d_elements_iter++;
            copyNodesToGPU(cpuElement->getRight(), d_elements_iter, d_interval_iter, next);

        }
    }



    __device__
    void buildShortcutGPU(DeviceElement* element, int i1, int i2, double totalAbsFlux, int shortcutSize, DeviceElement** shortcut) {
        if (i1 == i2) return;

        if (element->left != nullptr && element->right != nullptr) { // Check if it's a node
            double f = element->right->leftAbsFlux;
            int imid = int(f * shortcutSize / totalAbsFlux);
            if (imid < i1) {
                buildShortcutGPU(element->right, i1, i2, totalAbsFlux, shortcutSize, shortcut);
            } else if (imid >= i2) {
                buildShortcutGPU(element->left, i1, i2, totalAbsFlux, shortcutSize, shortcut);
            } else {
                shortcut[imid] = element;
                buildShortcutGPU(element->left, i1, imid, totalAbsFlux, shortcutSize, shortcut);
                buildShortcutGPU(element->right, imid + 1, i2, totalAbsFlux, shortcutSize, shortcut);
            }
        } else { // Leaf node
            for (int i = i1; i < i2; ++i) {
                shortcut[i] = element;
            }
        }
    }

    __global__
    void buildShortcutKernel(DeviceElement* elements, int numElements, double totalAbsFlux, int shortcutSize, DeviceElement** shortcut) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx == 0) {
            buildShortcutGPU(&elements[0], 0, shortcutSize, totalAbsFlux, shortcutSize, shortcut);
        }
    }


    void CuIntervalProbabilityTree::print_shortcut()
    {
        for (int i = 0; i < _shortcutSize; i++) {
            
            double absflux = 0. ;
            DeviceElement** element = _d_shortcut + i ;
            DeviceElement* ptr = nullptr;
            cudaMemcpy(&ptr, element, sizeof(DeviceElement*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&absflux, &ptr->absFlux, sizeof(double), cudaMemcpyDeviceToHost);
        
        }
    }

    void CuIntervalProbabilityTree::printf_root(const Element* root, DeviceElement* gpu_root) const
    {
        
        double absflux = 0. ;
        double leftabsflux = 0.;
        cudaMemcpy(&absflux, &gpu_root->absFlux, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&leftabsflux, &gpu_root->leftAbsFlux, sizeof(double), cudaMemcpyDeviceToHost);
        Device_Interval * data_ptr;
        cudaMemcpy(&data_ptr, &gpu_root->data, sizeof(Device_Interval*), cudaMemcpyDeviceToHost);
        
        if (root->getData().get() != nullptr)
        {
            Device_Interval* data_ptr; // 指向设备内存的指针
            cudaMemcpy(&data_ptr, &gpu_root->data, sizeof(Device_Interval*), cudaMemcpyDeviceToHost);
            double flux = 0. ;
            cudaMemcpy(&flux, &data_ptr->_flux, sizeof(double), cudaMemcpyDeviceToHost);
            // printf("root flux:%f, gpu flux: %f\n", root->getData()->getFlux(), flux);
           
        }
        if(root->isNode())
        {
            DeviceElement * left;
            DeviceElement * right;
            cudaMemcpy(&left, &gpu_root->left, sizeof(DeviceElement*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&right, &gpu_root->right, sizeof(DeviceElement*), cudaMemcpyDeviceToHost);
            printf_root(root->getLeft(), left);
            printf_root(root->getRight(), right);
        }
        
    }

    void CuIntervalProbabilityTree::CopyTreeToGpu()
    {

        // 1. 计算树中节点的数量
        std::vector<const Element*> allElements;
        getAllElements(this->_root, allElements);

         
        size_t numElements = allElements.size();
        _shortcutSize = _shortcut.size() ;
        // 2. 为设备端的元素数组分配内存
        CUDA_CHECK_RETURN(cudaMalloc((void**)&_d_elements, numElements * sizeof(DeviceElement)));
        CUDA_CHECK_RETURN(cudaMalloc((void**)&_d_interval, numElements * sizeof(Device_Interval)));           
        CUDA_CHECK_RETURN(cudaMalloc((void**)&_d_shortcut, _shortcutSize * sizeof(DeviceElement*)));

        CUDA_CHECK_RETURN(cudaMemset(_d_elements, 0, numElements * sizeof(DeviceElement)));
        CUDA_CHECK_RETURN(cudaMemset(_d_interval, 0, numElements * sizeof(Device_Interval)));
        CUDA_CHECK_RETURN(cudaMemset(_d_shortcut, 0, _shortcutSize * sizeof(DeviceElement*)));
              
        
        Device_Interval * interval_start = _d_interval;
        
        DeviceElement* gpuRoot = _d_elements;
        DeviceElement* currentGPUElement = gpuRoot;        

        DeviceElement * d_elements_iter = _d_elements;
        Device_Interval * d_interval_iter = _d_interval;

        d_elements_iter++;

        copyNodesToGPU(this->_root, d_elements_iter, d_interval_iter, currentGPUElement) ;


        Device_Interval host[numElements] = {0};
        CUDA_CHECK_RETURN(cudaMemcpy(host, interval_start, numElements * sizeof(Device_Interval), cudaMemcpyDeviceToHost));

        // printf_root(this->_root, gpuRoot);
        // 7. 复制快捷方式数组到设备端

        int blockSize = 1;
        int numBlocks = 1;
        buildShortcutKernel<<<numBlocks, blockSize>>>(gpuRoot, numElements, this->_totalAbsFlux, _shortcutSize, _d_shortcut);

        // print_shortcut();

    }

}


#endif