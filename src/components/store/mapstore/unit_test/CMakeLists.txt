cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(mapstore-tests CXX)

set(CMAKE_CXX_STANDARD 11)
set(GCC_COVERAGE_COMPILE_FLAGS "-g -O2 -fPIC")

include_directories(../../../../lib/common/include)
include_directories(../../../)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(mapstore-test1 test1.cpp)
target_link_libraries(mapstore-test1 ${ASAN_LIB} common numa gtest pthread dl)

