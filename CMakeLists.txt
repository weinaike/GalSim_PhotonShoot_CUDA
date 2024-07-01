cmake_minimum_required(VERSION 3.18)


# 说明文档
# 编译C++代码:
#   mkdir build; cd build; cmake ..; make; make install
# 编译包含cuda代码
#   mkdir build; cd build; cmake -DENABLE_CUDA=ON ..; make; make install

# 选项
#   ENABLE_CUDA: 是否编译CUDA代码
#   内部代码关联 宏Enable_CUDA

# 依赖库
#   Python: Python库
#   pybind11: pybind11库
#   Eigen3: Eigen3库 （镜像中安装）
#       其安装方法：sudo apt-get install libeigen3-dev
#   FFTW3: FFTW3库 (镜像中安装)
#       其安装方法：sudo apt-get install libfftw3-dev
#   CUDA: CUDA库 (映射到容器?)
#       需要生效nvcc，设置环境变量CUDACXX=/usr/local/cuda/bin/nvcc
#       CUDA安装目录：/usr/local/cuda-12.3

# 测试编译是否生效
#   1. 首先可编辑安装，  pip install -e . 
#   2. 修改src/SBGaussian.cpp， 生效 #define DEBUGLOGGING
#   3. 用以上方法编译库
#   4. python examples/demo1.py 
#   5. 查看输出结果， 会有大量的输出debug信息， 则说明库替换成功


set(LIB_NAME "_galsim.cpython-39-x86_64-linux-gnu")

OPTION(ENABLE_CUDA      "使能CUDA库"               OFF)

if (ENABLE_CUDA)
    project("GalSim" LANGUAGES CXX CUDA)
else()
    project("GalSim" LANGUAGES CXX)
endif()


set(CMAKE_C_COMPILER /usr/bin/gcc)
set(CMAKE_CXX_COMPILER /usr/bin/g++)


if (ENABLE_CUDA)
    # CUDA依赖配置
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DEnable_CUDA)
    if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
        link_directories(${CUDAToolkit_LIBRARY_DIRS})
    endif()

    set(CUDA_LIBS ${CUDA_LIBS} CUDA::cudart CUDA::cublas CUDA::curand)

    # 编译 cu 文件, 把所有的cu文件都编译成一个动态库
    file(GLOB_RECURSE CU_SRC "src/*.cu")

    foreach(f ${CU_SRC})
        message(STATUS "CU_SRC: ${f}")
    endforeach()

    add_library(cuda_lib SHARED ${CU_SRC})
    target_link_libraries(cuda_lib PRIVATE ${CUDA_LIBS})    

endif ()


# 查找Python库
find_package(Python COMPONENTS Interpreter Development NumPy)

if(Python_FOUND)
    message(STATUS "Found Python: ${Python_VERSION}")
    message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python_NumPy_INCLUDE_DIRS: ${Python_NumPy_INCLUDE_DIRS}")
    message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
    message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
    include_directories( ${PYTHON_INCLUDE_DIRS} ${Python_NumPy_INCLUDE_DIRS})
endif ()
# 查找pybind11库
find_package(pybind11 REQUIRED)
if(pybind11_FOUND)
    message(STATUS "Found pybind11: ${pybind11_VERSION}")
    include_directories(${pybind11_INCLUDE_DIRS})
endif()


# 查找EIGEN库
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_VERSION}")
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

find_package(FFTW3 REQUIRED)
if(FFTW3_FOUND)
    message(STATUS "Found FFTW3: ${FFTW3_VERSION}")
    include_directories(${FFTW3_INCLUDES})
    link_directories(${FFTW3_LIBRARY_DIRS})
endif()


# Galsim库头文件
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/galsim)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pthread -Wsign-compare -DNDEBUG -fPIC -fwrapv -O2 -Wall -std=c++11 -fvisibility=hidden -fopenmp -msse2")

# add_subdirectory(src)


# 编译galsim src c++ 文件
file(GLOB_RECURSE SOURCES "src/*.cpp"  "src/*.cc" "src/*.inst" )
list(REMOVE_ITEM SOURCES ${CMAKE_SOURCE_DIR}/src/mmgr.cpp)
# foreach(f ${SOURCES})
#     message(STATUS "SOURCES: ${f}")
# endforeach(f)
add_library(galsim STATIC ${SOURCES})




# add_subdirectory(pysrc)

file(GLOB PYSOURCES "pysrc/*.cpp")

add_library(cpython_galsim  SHARED  ${PYSOURCES})
# set_target_properties(cpython_galsim PROPERTIES LINK_FLAGS "-Wl,-rpath,${FFTW3_INCLUDES}")
target_link_libraries(cpython_galsim galsim ${FFTW3_LIBRARIES} pthread gomp ${PYTHON_LIBRARIES} ${pybind11_LIBRARIES}) 


# 设置动态库名称，与安装路径
set_target_properties(cpython_galsim PROPERTIES OUTPUT_NAME ${LIB_NAME}  PREFIX "")

install(TARGETS cpython_galsim LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/galsim)

# 测试目录
# file(GLOB SOURCES "tests/*.cpp")
# add_executable("galsim_test"  ${SOURCES})
# target_link_libraries(galsim_test galsim ${FFTW3_LIBRARIES} )