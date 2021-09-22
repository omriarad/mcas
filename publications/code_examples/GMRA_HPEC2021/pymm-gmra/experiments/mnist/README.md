# INSTRUCTIONS

Please make sure that you've built and installed the c++ code to whatever python environment
you're going to use. This can be done from the pymm-gmra directory by typing:
```<python_interpreter> setup.py install```

Then, please follow these steps:
Step 1) ```covertree_build.py``` This script takes one required argument and one optional argument. The required argument is the path to a directory. If that directory doesn't exist, the code will attempt to create it for you. In this directory is where the build covertree will serialize to (as a json file). The optional argument is whether or not to validate the tree after construction (expensive operation).

