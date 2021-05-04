#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#include "captured_ndarray.h"

int CapturedArrayObject_init(CapturedArrayObject *self, PyObject *args, PyObject *kwds)
{
  import_array();
  return 0;
}

PyObject * CapturedArrayObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  CapturedArrayObject *self;

  self = (CapturedArrayObject *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->x = 99;
    // self->first = PyUnicode_FromString("");
    // if (self->first == NULL) {
    //     Py_DECREF(self);
    //     return NULL;
    // }

    // self->last = PyUnicode_FromString("");
    // if (self->last == NULL) {
    //     Py_DECREF(self);
    //     return NULL;
    // }

    // self->number = 0;
  }

  return (PyObject *)self;
}

void CapturedArrayObject_destructor(PyObject *)
{
  return;
}



#pragma GCC diagnostic pop
