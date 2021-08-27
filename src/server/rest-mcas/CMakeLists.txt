cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(rest-mcas-server LANGUAGES CXX)

find_package(OpenSSL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-pedantic -Wall -Wextra -Werror -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wformat=2 -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)

add_compile_options(-DCONFIG_DEBUG -DRAPIDJSON_HAS_STDSTRING)

configure_file(src/rest_server_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/rest_server_config.h)

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/components)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/pistache/include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_INSTALL_PREFIX}/include)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

add_executable(${PROJECT_NAME} src/rest_server.cpp src/endpoint.cpp)

target_link_libraries(${PROJECT_NAME} common boost_program_options ${CMAKE_BINARY_DIR}/src/lib/pistache/src/libpistache-0.0.002-git20210409.so ssl crypto cityhash)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
