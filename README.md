# mcas
Memory Centric Active Storage - open source releases dropped to 'master' branch.  External contributions should be merged requested to the 'external' branch.


### update submodules
```bash
git submodule update --init --recursive
```

### configure
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_PYTHON_SUPPORT=1 -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
```

### one-time build
```bash
make bootstrap
```

### normal build
```bash
make -j
```


