# mcas
Memory Centric Active Storage (Private Repository)

Features from this repository may drop into Comanche for public release as needed.


## update submodules
```bash
git submodule update --init --recursive
```

## configure
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_PYTHON_SUPPORT=1 -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
```

# one-time build
```bash
make bootstrap
```

## normal build
```bash
make -j
```


