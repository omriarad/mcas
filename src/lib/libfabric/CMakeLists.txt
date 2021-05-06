cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraries, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# build libfabric
ExternalProject_Add(libfabric
  BUILD_IN_SOURCE 1
  UPDATE_DISCONNECTED 1 # will not update automatically
  GIT_REPOSITORY https://github.com/ofiwg/libfabric.git
  GIT_TAG v1.10.1
  CONFIGURE_COMMAND ./autogen.sh
  COMMAND ./configure --prefix=${CMAKE_INSTALL_PREFIX} $<$<CONFIG:Debug>:--enable-debug=yes> --enable-udp=no --enable-mlx=no --enable-sockets=yes --enable-verbs=yes
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  )

# Notes on building fabtest
#  cd src/lib/libfabric/libfabric-prefix/src/libfabric/fabtests
#  # in configure.ac, 'test -d $withval/lib64' is wrong for mcas: there is a lib64, but libfabric is not there.
#  sed -i 's?-d $withval/lib64?-f $withval/lib64/libfabric.so?' configure.ac
#  ./autogen.sh
#  ./configure --prefix=/home/clem/e/sv/debug/dist --with-libfabric=/home/clem/e/sv/debug/dist --enable-debug
#  make && make install
