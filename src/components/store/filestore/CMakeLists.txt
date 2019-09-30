cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory(./unit_test)

project(component-filestore CXX)

set (CMAKE_CXX_STANDARD 14)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

include_directories(../../../lib/common/include)
include_directories(../../)
include_directories(${CMAKE_INSTALL_PREFIX}/include) # tbb

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # libtbb_static
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # libtbb_static

file(GLOB SOURCES src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common numa dl rt boost_system boost_filesystem pthread tbb_static)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install (TARGETS ${PROJECT_NAME}
    LIBRARY
    DESTINATION lib)

