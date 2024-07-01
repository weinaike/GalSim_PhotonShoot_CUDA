file(GLOB SOURCES "*.cpp")
file(GLOB MATH_SOURCES "math/*.cpp")

list(APPEND SOURCES "hsm/PSFCorr.cpp" "Image.inst" ${MATH_SOURCES})
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/mmgr.cpp")

add_library("galsim" STATIC ${SOURCES})
