cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(component-mapstore CXX)

set (CMAKE_CXX_STANDARD 14)

add_subdirectory(./unit_test)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

include_directories(${CMAKE_INSTALL_PREFIX}/include) # city.h
include_directories(../../../lib/common/include)
include_directories(../../)

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # tbb tbbmalloc
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # tbb tbbmalloc

# for serious debugging
#add_compile_options("$<$<CONFIG:Debug>:-O0>")

file(GLOB SOURCES src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common numa dl rt boost_system pthread tbb tbbmalloc tbbmalloc_proxy)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install (TARGETS ${PROJECT_NAME}
    LIBRARY
    DESTINATION lib)

