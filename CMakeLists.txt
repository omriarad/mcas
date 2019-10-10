cmake_minimum_required(VERSION 3.5)

option(FORCE_OTT_BUILD "force out of tree build" ON)
option(BOOTSTRAP_DEPS "bootstrap to build deps" ON)
option(TEST_HSTORE_PERISHABLE "hstore perishable testing enabled" OFF)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)
    message(FATAL_ERROR "Cannot use in-source build ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}. You should delete any CMakeCache.txt and CMakeFiles and then try out-of-tree build")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)

set(GCC_COVERAGE_COMPILE_FLAGS "-g -shared-libgcc -DTBB_USE_THREADING_TOOLS")
set(ENV{MCAS_HOME} ${CMAKE_CURRENT_SOURCE_DIR})

include(CheckIncludeFiles)
find_package(PkgConfig)
include(${CMAKE_CURRENT_SOURCE_DIR}/mk/common.cmake)
project(comanche)

# locate FindXXX.cmake
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mk)
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# run 'make bootstrap' to  make comanche deps and core
# then run 'make' to build components/tests
add_custom_target(bootstrap
  bash ${CMAKE_SOURCE_DIR}/mk/bootstrap.sh ${CMAKE_SOURCE_DIR}
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

# bootstrap build should be performed one off
add_subdirectory(src)
add_subdirectory(testing)
add_subdirectory(examples)


