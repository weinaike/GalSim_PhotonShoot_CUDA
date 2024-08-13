#ifndef __CuProbabilityTree_h__
#define __CuProbabilityTree_h__


#include "Interval.h"
#include "ProbabilityTree.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "cuda_check.h"
namespace galsim {
   
    // 定义设备端的 Element 结构体
    struct DeviceElement {
        double absFlux;
        double leftAbsFlux;
        // 添加其他必要的成员变量
        DeviceElement* left;
        DeviceElement* right;
        Device_Interval* data;
    };

    class CuIntervalProbabilityTree : ProbabilityTree<Interval>
    {
        typedef typename std::vector<shared_ptr<Interval> >::iterator VecIter;

    public:
        using std::vector<shared_ptr<Interval> >::size;
        using std::vector<shared_ptr<Interval> >::begin;
        using std::vector<shared_ptr<Interval> >::end;
        using std::vector<shared_ptr<Interval> >::push_back;
        using std::vector<shared_ptr<Interval> >::insert;
        using std::vector<shared_ptr<Interval> >::empty;
        using std::vector<shared_ptr<Interval> >::clear;


    public:
        using ProbabilityTree<Interval>::buildTree; // 使基类的 buildTree 方法在派生类中可访问

        CuIntervalProbabilityTree() {this->_root = nullptr;}

        /// @brief Destructor - kill the `Element`s that have been stored away
        ~CuIntervalProbabilityTree() { 
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
            if (_d_interval != nullptr)
            {
                cudaFree(_d_interval);
                _d_interval = nullptr;
            } 

        }
   
        // 复制整个树到 GPU 的函数
        void CopyTreeToGpu();
        // 递归收集所有节点
        void getAllElements(const Element* root, std::vector<const Element*>& elements) {
            if (!root) return;
            // printf("%p\n",root->getData().get());
            elements.push_back(root);
            getAllElements(root->getLeft(), elements);
            getAllElements(root->getRight(), elements);
        }
        void copyNodesToGPU(const Element* cpuElement, DeviceElement * & d_elements_iter, Device_Interval * & d_interval_iter,  DeviceElement*& currentGPUElement) ;
        void print_shortcut();

        DeviceElement* getDeviceElements() const { return _d_elements; }
        DeviceElement** getDeviceShortcut() const { return _d_shortcut; }      
        void find_and_interpolateFlux(long seed, double * x, double* y, double* flux, 
            int N, double fluxPerPhoton, const bool isRadial, bool xandy) const;

        void printf_root(const Element* root, DeviceElement* gpu_root) const;

    private:       
        // GPU 端的指针
        DeviceElement* _d_elements = nullptr;
        DeviceElement** _d_shortcut = nullptr;
        int _shortcutSize;
        Device_Interval* _d_interval = nullptr;

    };

}
#endif  // ENABLE_CUDA
#endif  // __CuProbabilityTree_h__