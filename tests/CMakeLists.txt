file(GLOB SOURCES "*.cpp")

find_package(numpy)

add_executable("galsim_test"  ${SOURCES})
target_link_libraries(galsim_test galsim fftw3)

