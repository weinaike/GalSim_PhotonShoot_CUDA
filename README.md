# GalSim CUDA 加速版本使用说明

## 项目介绍

本项目是针对 GalSim 库的一项优化改进，专注于光子射击渲染方法的 CUDA 加速版本开发。通过将渲染方法迁移到 CUDA，使其能够在 NVIDIA 显卡上加速运行，从而显著提升计算效率。代码迁移过程中，保持了对外接口的一致性，因此项目既支持 C++ 实现，也支持 CUDA 实现。通过启用 ENABLE_CUDA 编译选项，可以选择激活 CUDA 加速。

## 安装说明

### 环境准备

#### 系统要求:

* 操作系统: Linux  (需安装 NVIDIA 驱动)
* CUDA Toolkit 版本: >= 11.0

#### 依赖工具:

* CMake (>= 3.15)
* Python (>= 3.9)
* Pip

#### 硬件要求:

* 支持 CUDA 的 NVIDIA 显卡

### 安装步骤

1. 本地可编辑安装

由于项目需要支持版本切换，建议采用本地可编辑安装方式：

```shell
pip install . -e
```

执行上述命令后，将完成基于 C++ 的 GalSim 库的安装。

2. 启用 CUDA 版本

如果需要切换到 CUDA 实现，需要手动编译 CUDA 代码和 C++ 的共享库。以下是具体步骤：

```shell
# 创建构建目录并进入：
mkdir build
cd build
#配置项目，启用 CUDA：
cmake .. -DENABLE_CUDA=ON
#编译代码：
make -j12
#提示：-j12 指定使用 12 个线程进行并行编译，根据设备性能可调整线程数。
#安装编译生成的库：
make install
```

执行以上步骤后，将完成 CUDA 版本的_galsim.cpython-39-x86_64-linux-gnu.so库更换。

注意：_galsim.cpython-39-x86_64-linux-gnu.so的名称英文python版本不同而不同。可以通过修改CMakeLists.txt中的LIB_NAME实现。

### 注意事项

* **版本切换:** 若需要从 CUDA 版本切换回 C++ 版本，请重新编译并确保 ENABLE_CUDA 未启用。
* **性能优化:** 确保显卡驱动与 CUDA Toolkit 版本匹配，以发挥最佳性能。使用 nvidia-smi 检查 GPU 资源占用情况。
* **问题排查:** 如果编译或运行中出现问题，请检查 CMake 日志和 CUDA 编译器输出。 确保 Python 环境干净，避免依赖冲突。

## 支持与反馈

如果在使用过程中遇到问题或有改进建议，请通过以下方式联系我们：

项目主页: [GitHub 链接](https://github.com/weinaike/GalSim_PhotonShoot_CUDA)

邮件: weinaike@zhejianglab.org

感谢您对 GalSim CUDA 加速版本的支持！
