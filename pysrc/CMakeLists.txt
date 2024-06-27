file(GLOB PYSOURCES "*.cpp")


add_library("cpython_galsim"  SHARED  ${PYSOURCES})
target_link_libraries(cpython_galsim galsim fftw3 pthread gomp) 
set_target_properties(cpython_galsim PROPERTIES LINK_FLAGS "-Wl,-rpath,${FFTW3_LIB_DIR}")
