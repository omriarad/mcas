cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(fabric-tests CXX)

# TODO fix to absolute
include_directories(../../../../components)
include_directories(../../../../lib/common/include/)
include_directories(../../../../lib/GSL/include/)
include_directories(../../../../lib/cw/include/)

include_directories(${CMAKE_INSTALL_PREFIX}/include) # daxctl, tbb

link_directories(${CMAKE_INSTALL_PREFIX}/lib) # libfabric
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(GTEST_LIB gtest)
else()
  set(GTEST_LIB gtestd)
endif()

add_executable(fabric-test1 test1.cpp
  eyecatcher.cpp
  mr_lock.cpp
  patience.cpp
  pingpong_buffer_state.cpp
  pingpong_cb_ctxt.cpp
  pingpong_cb_pack.cpp
  pingpong_client.cpp
  pingpong_cnxn_state.cpp
  pingpong_server.cpp
  pingpong_server_cb.cpp
  pingpong_server_client_state.cpp
  pingpong_server_n.cpp
  registered_memory.cpp
  registration.cpp
  remote_memory_accessor.cpp
  remote_memory_client.cpp
  remote_memory_client_grouped.cpp
  remote_memory_server.cpp
  remote_memory_server_grouped.cpp
  remote_memory_subclient.cpp
  remote_memory_subserver.cpp
  server_connection_and_memory.cpp
  server_connection.cpp
  server_grouped_connection.cpp
  wait_poll.cpp
)

target_link_libraries(fabric-test1 ${ASAN_LIB} ${GTEST_LIB} common numa fabric pthread dl)

add_executable(fabric-test2 test2.cpp
  eyecatcher.cpp
  mr_lock.cpp
  patience.cpp
  pingpong_buffer_state.cpp
  pingpong_cb_ctxt.cpp
  pingpong_cb_pack.cpp
  pingpong_client.cpp
  pingpong_cnxn_state.cpp
  pingpong_server.cpp
  pingpong_server_cb.cpp
  pingpong_server_client_state.cpp
  pingpong_server_n.cpp
  registered_memory.cpp
  registration.cpp
  remote_memory_accessor.cpp
  remote_memory_client.cpp
  remote_memory_client_grouped.cpp
  remote_memory_server.cpp
  remote_memory_server_grouped.cpp
  remote_memory_subclient.cpp
  remote_memory_subserver.cpp
  server_connection_and_memory.cpp
  server_connection.cpp
  server_grouped_connection.cpp
  wait_poll.cpp
)
target_link_libraries(fabric-test2 ${ASAN_LIB} ${GTEST_LIB} common numa fabric pthread dl)

add_executable(fabric-test3 test3.cpp
  eyecatcher.cpp
  mr_lock.cpp
  patience.cpp
  pingpong_buffer_state.cpp
  pingpong_cb_ctxt.cpp
  pingpong_cb_pack.cpp
  pingpong_client.cpp
  pingpong_cnxn_state.cpp
  pingpong_server.cpp
  pingpong_server_cb.cpp
  pingpong_server_client_state.cpp
  pingpong_server_n.cpp
  registered_memory.cpp
  registration.cpp
  server_connection.cpp
  wait_poll.cpp
)
target_link_libraries(fabric-test3 ${ASAN_LIB} ${GTEST_LIB} common numa fabric pthread dl)

add_executable(fabric-testbw testbw.cpp
  eyecatcher.cpp
  patience.cpp
  registered_memory.cpp
  registration.cpp
  remote_memory_accessor.cpp
  remote_memory_client.cpp
  remote_memory_server.cpp
  server_connection_and_memory.cpp
  server_connection.cpp
  wait_poll.cpp
)
target_link_libraries(fabric-testbw ${ASAN_LIB} ${GTEST_LIB} common numa fabric pthread dl)
set_target_properties(fabric-testbw PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

add_executable(fabric-testcw cw.cpp
  eyecatcher.cpp
  registered_memory.cpp
  registration.cpp
  server_connection.cpp
)
target_link_libraries(fabric-testcw ${ASAN_LIB} common numa fabric pthread dl)
set_target_properties(fabric-testcw PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

add_executable(fabric-testcw_client cw_client.cpp)
target_link_libraries(fabric-testcw_client ${ASAN_LIB} common numa fabric pthread dl cw)
set_target_properties(fabric-testcw_client PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

add_executable(fabric-testcw_server cw_server.cpp)
target_link_libraries(fabric-testcw_server ${ASAN_LIB} common numa fabric nupm pthread dl cw)
set_target_properties(fabric-testcw_server PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

install(TARGETS fabric-test1 fabric-test2 fabric-test3 fabric-testbw fabric-testcw fabric-testcw_client fabric-testcw_server DESTINATION testing)
install(FILES "bw-client.sh" "bw.sh" "cw-client.sh" "cw.sh" DESTINATION testing
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
