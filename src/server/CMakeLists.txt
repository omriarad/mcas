cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

if(BOOTSTRAP_DEPS)
else(BOOTSTRAP_DEPS)
add_subdirectory(mcas)
add_subdirectory(ado)
endif(BOOTSTRAP_DEPS)

