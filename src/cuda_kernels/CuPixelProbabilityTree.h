#ifndef __CuPixelProbabilityTree_h__
#define __CuPixelProbabilityTree_h__


#include "ProbabilityTree.h"
#include "SBInterpolatedImage.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "cuda_check.h"

namespace galsim {

    struct Device_Pixel
    {
        double x;
        double y;
        bool isPositive;
        double flux;
    };
    // 定义设备端的 Element 结构体

    struct DevicePixelElement {
        double absFlux;
        double leftAbsFlux;
        // 添加其他必要的成员变量
        DevicePixelElement* left;
        DevicePixelElement* right;
        
        Device_Pixel * data;
    };

    class CuPixelProbabilityTree : ProbabilityTree<Pixel>
    {
        typedef typename std::vector<shared_ptr<Pixel> >::iterator VecIter;

    public:
        using std::vector<shared_ptr<Pixel> >::size;
        using std::vector<shared_ptr<Pixel> >::begin;
        using std::vector<shared_ptr<Pixel> >::end;
        using std::vector<shared_ptr<Pixel> >::push_back;
        using std::vector<shared_ptr<Pixel> >::insert;
        using std::vector<shared_ptr<Pixel> >::empty;
        using std::vector<shared_ptr<Pixel> >::clear;


    public:
        using ProbabilityTree<Pixel>::buildTree; // 使基类的 buildTree 方法在派生类中可访问

        CuPixelProbabilityTree() {this->_root = nullptr;}

        /// @brief Destructor - kill the `Element`s that have been stored away
        ~CuPixelProbabilityTree() { 
            if (_root != nullptr)
            {
                delete _root; 
                _root = nullptr;
            } 
            // 释放 GPU 端的内存//有问题
            if (_d_elements != nullptr)
            {
                cudaFree(_d_elements);
                _d_elements = nullptr;
            } 
            if (_d_shortcut != nullptr) 
            {
                cudaFree(_d_shortcut);
                _d_shortcut = nullptr;
            }
            if (_d_pixel != nullptr)
            {
                cudaFree(_d_pixel);
                _d_pixel = nullptr;
            }
        }
   
        // 复制整个树到 GPU 的函数
        void CopyPixelTreeToGpu();
        // 递归收集所有节点
        void getAllElements(const Element* root, std::vector<const Element*>& elements) const {
            if (!root) return;
            // printf("%p\n",root->getData().get());
            elements.push_back(root);
            getAllElements(root->getLeft(), elements);
            getAllElements(root->getRight(), elements);
        }
        void copyNodesToGPU(const Element* cpuElement, DevicePixelElement * & d_elements_iter,
                Device_Pixel *& d_pixel_iter, DevicePixelElement*& currentGPUElement) const ;
        void print_shortcut() const ;

        DevicePixelElement* getDeviceElements() const { return _d_elements; }
        DevicePixelElement** getDeviceShortcut() const { return _d_shortcut; }      
        void find(long seed, double * x, double* y, double* flux, int N, double fluxPerPhoton) const;

        void printf_root(const Element* root, DevicePixelElement* gpu_root) const;
    private:       
        // GPU 端的指针
        DevicePixelElement* _d_elements = nullptr;
        DevicePixelElement** _d_shortcut = nullptr;
        int _shortcutSize;
        Device_Pixel* _d_pixel = nullptr;
    };

}
#endif
#endif