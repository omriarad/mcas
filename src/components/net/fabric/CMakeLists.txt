cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(component-fabric CXX)

add_compile_options("$<$<CONFIG:Debug>:-O0>")
add_compile_options("$<$<CONFIG:Release>:-DNDEBUG>")

add_subdirectory(./unit_test)

add_definitions(-DCONFIG_DEBUG) # P{LOG,DEG,INF,WRN,ERR} control

include_directories(${CMAKE_INSTALL_PREFIX}/include) # rdma
include_directories(../../../lib/common/include/)
include_directories(../../../components)
link_directories(${CMAKE_INSTALL_PREFIX}/lib) # libfabric

set(SOURCES
	src/addrinfo.cpp
	src/event_registration.cpp
	src/fabric_bad_alloc.cpp
	src/fabric_check.cpp
	src/fabric_client.cpp
	src/fabric_client_grouped.cpp
	src/fabric_comm_grouped.cpp
	src/fabric_connection_client.cpp
	src/fabric_connection.cpp
	src/fabric_connection_server.cpp
	src/fabric.cpp
	src/fabric_cq.cpp
	src/fabric_cq_generic_grouped.cpp
	src/fabric_cq_grouped.cpp
	src/fabric_endpoint.cpp
	src/fabric_endpoint_server.cpp
	src/fabric_factory.cpp
	src/fabric_generic_grouped.cpp
	src/fabric_json.cpp
	src/fabric_runtime_error.cpp
	src/fabric_server.cpp
	src/fabric_server_factory.cpp
	src/fabric_server_generic_factory.cpp
	src/fabric_server_grouped.cpp
	src/fabric_server_grouped_factory.cpp
	src/fabric_str.cpp
	src/fabric_util.cpp
	src/fabric_util_wait.cpp
	src/fd_control.cpp
	src/fd_pair.cpp
	src/fd_socket.cpp
	src/fd_unblock_set_monitor.cpp
	src/hints.cpp
	src/open_cnxns.cpp
	src/pending_cnxns.cpp
	src/system_fail.cpp
)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

target_compile_options(${PROJECT_NAME} PUBLIC -fPIC)
target_link_libraries(${PROJECT_NAME} common fabric pthread)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install (TARGETS ${PROJECT_NAME}
    LIBRARY
    DESTINATION lib)
