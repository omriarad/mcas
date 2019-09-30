cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(component-hstore CXX)

set (CMAKE_CXX_STANDARD 14)

add_subdirectory(./unit_test)

enable_language(CXX)
add_compile_options(-g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)
# add_compile_options("$<$<CONFIG:Debug>:-O0>")

set(SOURCES src/dax_map.cpp src/heap_rc.cpp src/hop_hash_log.cpp src/hstore.cpp src/hstore_factory.cpp src/hop_hash.cpp src/owner.cpp src/perishable.cpp)

#add_custom_target(format-hstore
#  COMMAND /usr/bin/clang-format
#  -style=file
#  -i
#  ${ALL_SOURCES})

# if(TEST_HSTORE_PERISHABLE)
#  add_compile_options(-DTEST_HSTORE_PERISHABLE=1)
# endif()

get_property(incdirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(sysincludes "-I/usr/include/c++/5 -I/usr/include/x86_64-linux-gnu/c++/5/ -I/usr/include/linux")
# add -I prefix
string(REGEX REPLACE "([^;]+)" "-I\\1" istring "${incdirs}")

include_directories(${CMAKE_SOURCE_DIR}/src/lib/libpmem/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(../../../lib/common/include)
include_directories(../../)

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # cityhash tbb tbbmalloc
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # cityhash tbb tbbmalloc

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
endif()
add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

target_compile_options(${PROJECT_NAME} PUBLIC "$<$<BOOL:${TEST_HSTORE_PERISHABLE}>:-DTEST_HSTORE_PERISHABLE=1>")
target_link_libraries(${PROJECT_NAME} common pthread numa dl rt boost_system boost_filesystem tbb nupm cityhash)

add_dependencies(${PROJECT_NAME} common nupm)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

