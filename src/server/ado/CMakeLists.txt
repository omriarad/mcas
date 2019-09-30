cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project (ado)

set(CMAKE_CXX_STANDARD 14)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
add_compile_options(-g -Wall -Wextra -Wcast-align -Wcast-qual -Wconversion -Wredundant-decls -Wshadow -Wtype-limits -Wno-unused-parameter -Wwrite-strings -Wno-variadic-macros )
#add_compile_options(-g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libnupm/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_SOURCE_DIR}/src/components/ado/ado_proxy/src)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libadoproto/generated)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/libadoproto/src)
include_directories(${CMAKE_INSTALL_PREFIX}/include) # docker.h
link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # flatbuffers tbb (via nupm) tbbmalloc (via nupm)
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # tbb (via nupm) tbbmalloc (via nupm)

add_executable(ado src/ado.cpp)

target_link_libraries(ado pthread boost_program_options nupm ado-proto)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

add_dependencies(${PROJECT_NAME} nupm)

install(TARGETS ado RUNTIME DESTINATION bin)
