cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory(lib)

if(BOOTSTRAP_DEPS)
else(BOOTSTRAP_DEPS)
add_subdirectory(kernel/modules)
add_subdirectory(components)
add_subdirectory(server)
add_subdirectory(python)
add_subdirectory(apps)
add_subdirectory(ado/EASTL)
endif(BOOTSTRAP_DEPS)
