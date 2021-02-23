cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(pmem C)
enable_language(C)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message(FATAL ${ARCHITECTURE})

find_library(LIB_NDCTL libndctl.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("./")
include_directories("./x86_64")
include_directories("./include")
include_directories("./common")
include_directories("../common/include")
include_directories(${CMAKE_INSTALL_PREFIX}/include) # ndctl

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # tbb tbbmalloc
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # tbb tbbmalloc

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

add_definitions(-DCONFIG_DEBUG)
TRY_RUN(RUN_RESULT_VAR COMPILE_RESULT_VAR "${CMAKE_INSTALL_PREFIX}/bin" "${CMAKE_CURRENT_SOURCE_DIR}/avx512f_test.c" COMPILE_DEFINITIONS "-DCMAKE_C_FLAGS=mavx512f")

add_compile_options(-mavx)

file(GLOB SOURCES libpmem.c memops_generic.c pmem.c pmem_posix.c ./common/*.c ./x86_64/cpu.c ./x86_64/init.c ./x86_64/memcpy/memcpy_*_sse2_*.c ./x86_64/memset/memset_*_sse2_*.c ./x86_64/memcpy/memcpy_*_avx_*.c ./x86_64/memset/memset_*_avx_*.c )

add_library(pmem SHARED ${SOURCES})

# if CPU supports avx512f, add that option and avx512f functions
if(RUN_RESULT_VAR AND COMPILE_RESULT_VAR)
add_compile_options(-mavx512f)
file(GLOB SOURCES ./x86_64/memcpy/memcpy_*_avx512f_*.c ./x86_64/memset/memset_*_avx512f_*.c)
endif()

#target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:-O0>")
target_link_libraries(${PROJECT_NAME} stdc++fs common numa pthread ${LIB_NDCTL} dl gcov)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
