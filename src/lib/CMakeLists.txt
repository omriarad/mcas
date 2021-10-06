cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# throw exception specification, used in tbb, disallowed as of C++17
set(CMAKE_CXX_STANDARD 14)

if(BOOTSTRAP_DEPS)
  set(BUILD_SHARED_LIBS ON)
  add_subdirectory(googletest)
  add_subdirectory(ndctl)
  add_subdirectory(rapidjson)
  add_subdirectory(cityhash)
  add_subdirectory(libbase64)
  add_subdirectory(common)
  add_subdirectory(libnupm/include/nupm)
  add_subdirectory(EASTL)
  add_subdirectory(GSL)
else(BOOTSTRAP_DEPS)
  add_compile_options($<$<BOOL:${CODE_COVERAGE}>:--coverage>)

  add_subdirectory(common)
  add_subdirectory(libpmem)
  add_subdirectory(libnop)
  add_subdirectory(libnupm)
  add_subdirectory(libccpm)
  add_subdirectory(libmm)
endif(BOOTSTRAP_DEPS)


