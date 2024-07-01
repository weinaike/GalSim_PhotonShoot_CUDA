file(GLOB SOURCES "*.cpp")

add_executable("galsim_test"  ${SOURCES})
target_link_libraries(galsim_test galsim fftw3)

