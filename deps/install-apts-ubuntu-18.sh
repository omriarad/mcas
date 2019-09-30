#!/bin/bash
#
# Packages for Ubuntu 18.04 LTS
#
apt-get update

apt-get install -y build-essential git cmake libnuma-dev libelf-dev libpcap-dev uuid-dev \
        sloccount doxygen libnuma-dev libcunit1 libcunit1-dev pkg-config \
        libboost-system-dev libboost-iostreams-dev libboost-program-options-dev \
        libboost-filesystem-dev libboost-date-time-dev libaio-dev \
        libssl-dev g++-multilib fabric libtool-bin autoconf automake libibverbs-dev librdmacm-dev \
        libpcap-dev libomp-dev \
        libboost-python-dev libkmod-dev libjson-c-dev libbz2-dev \
        libelf-dev libsnappy-dev liblz4-dev \
        asciidoc xmlto libtool libgtest-dev python3-numpy libudev-dev \
	      libgoogle-perftools-dev google-perftools libcurl4-openssl-dev linux-headers-generic

# special handling of googletest
#
cd /usr/src/googletest/googletest
mkdir build
cd build
cmake ..
make
cp libgtest* /usr/lib/
cd ..
rm -rf build
mkdir /usr/local/lib/googletest
ln -s /usr/lib/libgtest.a /usr/local/lib/googletest/libgtest.a
ln -s /usr/lib/libgtest_main.a /usr/local/lib/googletest/libgtest_main.a
