


#include "CuPixelProbabilityTree.h"
#include "ProbabilityTree.h"
#include <curand_kernel.h>

namespace galsim {



 // CUDA 内核函数，用于生成均匀分布在单位圆内的点
    __global__ void pixel_rand_shoot_kernel(long seed, double * x, double* y, double* flux, int N, double fluxPerPhoton,  // output
                        DevicePixelElement** d_shortcut, int shortcutSize, double totalAbsFlux  // find
                        ) 
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N) {
            curandState state;
            curand_init(seed, idx, 0, &state);

            double unitRandom = curand_uniform(&state) ;

            // find 
            int i = int(unitRandom * shortcutSize);
            DevicePixelElement* element = d_shortcut[i];
            unitRandom *= totalAbsFlux;

            // 用栈来模拟递归
            while (element->left || element->right) {
                if (unitRandom < element->right->leftAbsFlux) {
                    element = element->left;
                } else {
                    element = element->right;
                }
            }
            Device_Pixel * p =  element->data;
                        
            x[idx] = p->x;
            y[idx] = p->y;
            flux[idx] = p->isPositive ? fluxPerPhoton : -fluxPerPhoton;
        }
    }

    void CuPixelProbabilityTree::copyNodesToGPU(const Element* cpuElement, DevicePixelElement * & d_elements_iter, 
            Device_Pixel * & d_pixel_iter, DevicePixelElement*& currentGPUElement) const
    {
        double absflux =  cpuElement->getAbsFlux();
        double leftabsflux = cpuElement->getLeftAbsFlux();
        CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->leftAbsFlux, &leftabsflux, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->absFlux, &absflux, sizeof(double), cudaMemcpyHostToDevice));

        if (cpuElement->isLeaf()) {

            DevicePixelElement* next = nullptr;
            CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->left, &next, sizeof(DevicePixelElement*), cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->right, &next, sizeof(DevicePixelElement*), cudaMemcpyHostToDevice));       
            
            
            Pixel * ptr = cpuElement->getData().get(); 
            Device_Pixel host;
            host.x = ptr->x;
            host.y = ptr->y;
            host.isPositive = ptr->isPositive;
            host.flux = ptr->flux;
            // printf("host flux: %f, x: %f, y:%f, positive:%d\n", host.flux, host.x, host.y, host.isPositive);
            
            Device_Pixel * dev = d_pixel_iter;
            CUDA_CHECK_RETURN(cudaMemcpy(dev, &host, sizeof(Device_Pixel), cudaMemcpyHostToDevice));
            CUDA_CHECK_RETURN(cudaMemcpy(&currentGPUElement->data, &dev, sizeof(Device_Pixel*), cudaMemcpyHostToDevice));
            d_pixel_iter++;


        } else {
            // 设置指针并递归拷贝左右子树

            // 设置指针
            DevicePixelElement* next = d_elements_iter;
            CUDA_CHECK_RETURN(cudaMemcpy(&(currentGPUElement->left), &next, sizeof(DevicePixelElement*), cudaMemcpyHostToDevice));

            d_elements_iter++;
            copyNodesToGPU(cpuElement->getLeft(), d_elements_iter, d_pixel_iter, next);            

            next = d_elements_iter;
            CUDA_CHECK_RETURN(cudaMemcpy(&(currentGPUElement->right), &next, sizeof(DevicePixelElement*), cudaMemcpyHostToDevice));

            d_elements_iter++;
            copyNodesToGPU(cpuElement->getRight(), d_elements_iter, d_pixel_iter, next);

        }
    }



    __device__
    void PixelbuildShortcutGPU(DevicePixelElement* element, int i1, int i2, double totalAbsFlux, int shortcutSize, DevicePixelElement** shortcut) {
       
        if (i1 == i2) return;

        // assert(i1 * totalAbsFlux/shortcutSize >= element->leftAbsFlux - 1.e-8);
        // assert(i2 * totalAbsFlux/shortcutSize <= element->leftAbsFlux + element->absFlux + 1.e-8);

        if (element->left != nullptr && element->right != nullptr) { // Check if it's a node
            double f = element->right->leftAbsFlux;
            int imid = int(f * shortcutSize / totalAbsFlux);
            if (imid < i1) {
                PixelbuildShortcutGPU(element->right, i1, i2, totalAbsFlux, shortcutSize, shortcut);
            } else if (imid >= i2) {
                PixelbuildShortcutGPU(element->left, i1, i2, totalAbsFlux, shortcutSize, shortcut);
            } else {
                shortcut[imid] = element;                
                PixelbuildShortcutGPU(element->left, i1, imid, totalAbsFlux, shortcutSize, shortcut);
                PixelbuildShortcutGPU(element->right, imid + 1, i2, totalAbsFlux, shortcutSize, shortcut);
            }
        } else { // Leaf node
            for (int i = i1; i < i2; ++i) {
                shortcut[i] = element;
            }
        }
    }

    __global__
    void PixelbuildShortcutKernel(DevicePixelElement* elements, int numElements, double totalAbsFlux, int shortcutSize, DevicePixelElement** shortcut) {
        if (threadIdx.x == 0 && blockIdx.x == 0) { // Only one thread executes this
            // Ensure indices are within bounds
            if (shortcutSize > 0 && elements != nullptr && shortcut != nullptr) {
                PixelbuildShortcutGPU(&elements[0], 0, shortcutSize, totalAbsFlux, shortcutSize, shortcut);
            } else {
                printf("Error: Invalid shortcutSize or null pointers\n");
            }
        }
    }


    // 复制整个树到 GPU 的函数
    void CuPixelProbabilityTree::CopyPixelTreeToGpu()
    {
        // 1. 计算树中节点的数量
        std::vector<const Element*> allElements;
        getAllElements(this->_root, allElements);

         
        size_t numElements = allElements.size();
        _shortcutSize = _shortcut.size() ;

        // printf("numElements: %d, _shortcutSize: %d\n", numElements, _shortcutSize);
        // printf("numElements: %d, shortcutSize: %d\n", numElements, shortcutSize);
        // 2. 为设备端的元素数组分配内存
        CUDA_CHECK_RETURN(cudaMalloc((void**)&_d_elements, numElements * sizeof(DevicePixelElement)));   
        CUDA_CHECK_RETURN(cudaMalloc((void**)&_d_pixel, numElements * sizeof(Device_Pixel)));                   
        CUDA_CHECK_RETURN(cudaMalloc((void**)&_d_shortcut, _shortcutSize * sizeof(DevicePixelElement*)));

        CUDA_CHECK_RETURN(cudaMemset(_d_elements, 0, numElements * sizeof(DevicePixelElement)));
        CUDA_CHECK_RETURN(cudaMemset(_d_pixel, 0, numElements * sizeof(Device_Pixel)));
        CUDA_CHECK_RETURN(cudaMemset(_d_shortcut, 0, _shortcutSize * sizeof(DevicePixelElement*)));

        // Device_Pixel * pixel_start = _d_pixel;              
        
        DevicePixelElement* gpuRoot = _d_elements;
        DevicePixelElement* currentGPUElement = gpuRoot;   

        DevicePixelElement* d_elements_iter = _d_elements;
        Device_Pixel * d_pixel_iter = _d_pixel;

        d_elements_iter++;

        copyNodesToGPU(this->_root, d_elements_iter, d_pixel_iter, currentGPUElement) ;
       
        // printf_root(this->_root, gpuRoot);
        // 7. 堆栈大小设置，因为这个问题， 查了一天。

        size_t stackSize;
        CUDA_CHECK_RETURN(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
        size_t new_stack_size = 65536;  // 设定栈大小为 8192 字节
        CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitStackSize, new_stack_size));
        // printf("stack size: %d\n", stackSize);

        PixelbuildShortcutKernel<<<1, 1>>>(gpuRoot, numElements, this->_totalAbsFlux, _shortcutSize, _d_shortcut);
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());   
        CUDA_CHECK_RETURN(cudaDeviceSetLimit(cudaLimitStackSize, stackSize));
        // print_shortcut() ;
    }


    void CuPixelProbabilityTree::print_shortcut() const
    {

        for (int i = 0; i < _shortcutSize; i++) {
            
            double absflux = 0. ;
            DevicePixelElement** element = _d_shortcut + i ;
            DevicePixelElement* ptr = nullptr;
            cudaMemcpy(&ptr, element, sizeof(DevicePixelElement*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&absflux, &ptr->absFlux, sizeof(double), cudaMemcpyDeviceToHost);
        
        }
    }




    void CuPixelProbabilityTree::find(long seed, double * x, double* y, double* flux, int N, double fluxPerPhoton) const
    {
        time_t start, end;
        start = clock();

        int blockSize = 256; // Example block size
        int numBlocks = (N + blockSize - 1) / blockSize;

        pixel_rand_shoot_kernel<<<numBlocks, blockSize>>>(seed, x, y, flux, N, fluxPerPhoton,  _d_shortcut, _shortcutSize, this->_totalAbsFlux);        
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());
        CUDA_CHECK_RETURN(cudaGetLastError());        

        end = clock();
        double time = (double)(end - start) / CLOCKS_PER_SEC * 1000;
        // printf("find_and_interpolateFlux time: %f ms,    %d\n", time, N);
    }

    void CuPixelProbabilityTree::printf_root(const Element* root, DevicePixelElement* gpu_root) const
    {
        double absflux = 0. ;
        double leftabsflux = 0.;
        cudaMemcpy(&absflux, &gpu_root->absFlux, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&leftabsflux, &gpu_root->leftAbsFlux, sizeof(double), cudaMemcpyDeviceToHost);
        Device_Pixel * data_ptr;
        cudaMemcpy(&data_ptr, &gpu_root->data, sizeof(Device_Pixel*), cudaMemcpyDeviceToHost);
        printf("host ip : %p, gpu ip: %p\n", root, gpu_root);
        if (root->getData().get() != nullptr)
        {
            DevicePixelElement host = {0};
            cudaMemcpy(&host, gpu_root, sizeof(DevicePixelElement), cudaMemcpyDeviceToHost);
            printf("root absflux:%f, gpu absflux: %f, _totalAbsFlux: %f \n", root->getAbsFlux(), host.absFlux, this->_totalAbsFlux);
            printf("root leftflux:%f, gpu leftflux: %f, _totalAbsFlux:%f \n", root->getLeftAbsFlux(), host.leftAbsFlux, this->_totalAbsFlux);


            Device_Pixel* data_ptr; // 指向设备内存的指针
            cudaMemcpy(&data_ptr, &gpu_root->data, sizeof(Device_Pixel*), cudaMemcpyDeviceToHost);
            Device_Pixel pix = {0};
            cudaMemcpy(&pix, data_ptr, sizeof(Device_Pixel), cudaMemcpyDeviceToHost);
            printf("root pix flux :%f, gpu pix flux: %f | root x: %f , gpu x:%f\n", root->getData()->getFlux(), pix.flux, root->getData()->x, pix.x);    

            printf("left: %p, right: %p, gpu left:%p, right:%p\n", root->getLeft(), root->getRight(), host.left, host.right);


        }
        if(root->isNode())
        {
            DevicePixelElement * left;
            DevicePixelElement * right;
            cudaMemcpy(&left, &gpu_root->left, sizeof(DevicePixelElement*), cudaMemcpyDeviceToHost);
            cudaMemcpy(&right, &gpu_root->right, sizeof(DevicePixelElement*), cudaMemcpyDeviceToHost);

            printf("left: %p, right: %p\n", left, right);

            //
            DevicePixelElement host = {0};
            cudaMemcpy(&host, left, sizeof(DevicePixelElement), cudaMemcpyDeviceToHost);

            printf("left absflux:%f, gpu absflux: %f\n", root->getLeft()->getAbsFlux(), host.absFlux);
            printf_root(root->getLeft(), left);
            printf_root(root->getRight(), right);
        }
        
    }



}
