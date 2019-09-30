#!/bin/bash
apt-get update
apt-get install -y wget cmake build-essential git gcc make libcunit1-dev pkg-config libtool \
        libaio-dev libssl-dev libibverbs-dev librdmacm-dev libudev-dev libuuid1 uuid uuid-dev\
        libnuma-dev libaio-dev libcunit1 libcunit1-dev \
        libboost-system-dev libboost-iostreams-dev libboost-program-options-dev \
        libboost-filesystem-dev libboost-date-time-dev \
        libssl-dev g++-multilib fabric libtool-bin autoconf automake \
        libomp-dev libboost-python-dev libkmod-dev libjson-c-dev libbz2-dev \
        libelf-dev libsnappy-dev liblz4-dev \
        asciidoc xmlto \
        google-perftools libgoogle-perftools-dev libgtest-dev \
	      python-numpy libcurl4-openssl-dev linux-headers-generic

find /usr/src/gtest

# special handling of googletest
#
cd /usr/src/gtest
mkdir build
cd build
cmake ..
make
make install
cp libgtest* /usr/lib/
cd ..
rm -rf build
mkdir /usr/local/lib/googletest
ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a
ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a
