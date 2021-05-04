#ifndef __CAPTURED_NDARRAY_H__
#define __CAPTURED_NDARRAY_H__

#include <Python.h>
#include <numpy/ndarrayobject.h>

typedef struct {
  PyArrayObject base;
  int x;
} CapturedArrayObject;




#endif // __CAPTURED_NDARRAY_H__
