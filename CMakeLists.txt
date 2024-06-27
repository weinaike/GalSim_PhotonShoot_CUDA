cmake_minimum_required(VERSION 3.18)
project("GalSim")
set(CMAKE_C_COMPILER /usr/bin/gcc)

# NOTE: Add your configurations here
set(EIGEN_HOME_DIR  "downloaded_eigen/eigen-3.4.0" CACHE PATH "path to eigen")
set(PYTHON_INCL_DIR "/home/jiangbo/anaconda3/envs/autogen/include/python3.10" CACHE PATH "path to python header files")
set(PYBIND_INCL_DIR "/home/jiangbo/anaconda3/envs/autogen/lib/python3.10/site-packages/pybind11/include" CACHE PATH "path to pybind header files")
set(NUMPY_INCL_DIR  "/home/jiangbo/anaconda3/envs/autogen/lib/python3.10/site-packages/numpy/_core/include" CACHE PATH "path to numpy header files")
set(FFTW3_LIB_DIR   "/lib/x86_64-linux-gnu/" CACHE PATH "path to libfftw3")

include_directories("include" "include/galsim" "${EIGEN_HOME_DIR}" "${PYTHON_INCL_DIR}" "${PYBIND_INCL_DIR}" "${NUMPY_INCL_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -pthread -Wsign-compare -DNDEBUG -fPIC -fwrapv -O2 -Wall -std=c++11 -fvisibility=hidden -fopenmp -msse2")

add_subdirectory(src)
add_subdirectory(pysrc)
add_subdirectory(tests)
