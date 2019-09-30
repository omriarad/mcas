cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(ado-proto CXX)

set (CMAKE_CXX_STANDARD 14)
add_compile_options(-g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)

find_program(FLATC NAMES flatc PATHS "${CMAKE_INSTALL_PREFIX}/bin")

if (NOT FLATC)
  message(FATAL_ERROR ": Flatc compiler not found")
endif()

include_directories(${CMAKE_SOURCE_DIR}/src/lib/common/include)
include_directories(${CMAKE_SOURCE_DIR}/src/lib/flatbuffers/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/generated/)

link_directories(${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}) # flatbuffers

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/generated/ado_proto_generated.h
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/ado_proto.fbs
  COMMAND ${FLATC} --cpp -o ${CMAKE_CURRENT_SOURCE_DIR}/generated/ -c ${CMAKE_CURRENT_SOURCE_DIR}/src/ado_proto.fbs)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

set(SOURCES generated/ado_proto_generated.h src/ado_proto.cpp src/uipc.cpp src/uipc_channel.cpp src/uipc_shared_memory.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common numa dl rt boost_system pthread curl flatbuffers)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES
  PUBLIC_HEADER src/ado_proto.h
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install (TARGETS ${PROJECT_NAME}
  LIBRARY
  DESTINATION lib
  PUBLIC_HEADER DESTINATION include)

