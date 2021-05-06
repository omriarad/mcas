/*
   Copyright [2021] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#define PYMMCORE_API_VERSION "v0.1beta"
#define PAGE_SIZE 4096

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pymmcore_ARRAY_API

#include <Python.h>
#include <structmember.h>
#include <objimpl.h>
#include <pythread.h>
#include <numpy/arrayobject.h>

#include <api/components.h>
#include <api/kvstore_itf.h>

#if defined(__linux__)
#include <execinfo.h>
#endif

#include <list>
#include <common/logging.h>
#include "ndarray_helpers.h"

// forward declaration of custom types
//
extern PyTypeObject MemoryResourceType;

PyDoc_STRVAR(pymmcore_version_doc,
             "version() -> Get module version");
PyDoc_STRVAR(pymmcore_allocate_direct_memory_doc,
             "allocate_direct_memory(s) -> Returns 4K page-aligned memory view (experimental)");
PyDoc_STRVAR(pymmcore_free_direct_memory_doc,
             "free_direct_memory(s) -> Free memory previously allocated with allocate_direct_memory (experimental)");
PyDoc_STRVAR(pymcas_ndarray_header_size_doc,
             "ndarray_header_size(array) -> Return size of memory needed for header");
PyDoc_STRVAR(pymcas_ndarray_header_doc,
             "ndarray_header(array) -> Return ndarray persistent header");


static PyObject * pymmcore_version(PyObject * self,
                                   PyObject * args,
                                   PyObject * kwargs);

static PyObject * pymmcore_allocate_direct_memory(PyObject * self,
                                                  PyObject * args,
                                                  PyObject * kwargs);

static PyObject * pymmcore_free_direct_memory(PyObject * self,
                                              PyObject * args,
                                              PyObject * kwargs);


static PyMethodDef pymmcore_methods[] =
  {
   {"version",
    (PyCFunction) pymmcore_version, METH_NOARGS, pymmcore_version_doc },
   {"allocate_direct_memory",
    (PyCFunction) pymmcore_allocate_direct_memory, METH_VARARGS | METH_KEYWORDS, pymmcore_allocate_direct_memory_doc },
   {"free_direct_memory",
    (PyCFunction) pymmcore_free_direct_memory, METH_VARARGS | METH_KEYWORDS, pymmcore_free_direct_memory_doc },
   {"ndarray_header_size",
    (PyCFunction) pymcas_ndarray_header_size, METH_VARARGS | METH_KEYWORDS, pymcas_ndarray_header_size_doc },
   {"ndarray_header",
    (PyCFunction) pymcas_ndarray_header, METH_VARARGS | METH_KEYWORDS, pymcas_ndarray_header_doc },   
   {NULL, NULL, 0, NULL}        /* Sentinel */
  };


static PyModuleDef pymmcore_module = {
    PyModuleDef_HEAD_INIT,
    "pymmcore",
    "Python Micro MCAS module",
    -1,
    pymmcore_methods,
    NULL, NULL, NULL, NULL
};

                                     
PyMODINIT_FUNC
PyInit_pymmcore(void)
{  
  PyObject *m;

  PLOG("Init Pymm extension");

  import_array();

  MemoryResourceType.tp_base = 0; // no inheritance
  if(PyType_Ready(&MemoryResourceType) < 0) {
    assert(0);
    return NULL;
  }

  /* register module */
#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&pymmcore_module);
#else
#error "Extension for Python 3 only."
#endif

  if (m == NULL)
    return NULL;

  /* add types */
  int rc;

  Py_INCREF(&MemoryResourceType);
  rc = PyModule_AddObject(m, "MemoryResource", (PyObject *) &MemoryResourceType);
  if(rc) return NULL;

  return m;
}

/** 
 * Allocated memory view of aligned memory.  This memory
 * will not be garbage collected and should be explicitly 
 * freed
 * 
 * @param self 
 * @param args: size(size in bytes to allocate), zero(zero memory)
 * @param kwds 
 * 
 * @return memoryview object
 */
static PyObject * pymmcore_allocate_direct_memory(PyObject * self,
                                                  PyObject * args,
                                                  PyObject * kwds)
{
  static const char *kwlist[] = {"size",
                                 "zero",
                                 NULL};

  unsigned long nsize = 0;
  int zero_flag = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "k|p",
                                    const_cast<char**>(kwlist),
                                    &nsize,
                                    &zero_flag)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }
  
  char * ptr = static_cast<char*>(aligned_alloc(PAGE_SIZE, nsize));

  if(zero_flag)
    memset(ptr, 0x0, nsize);
  
  if(ptr == NULL) {
    PyErr_SetString(PyExc_RuntimeError,"aligned_alloc failed");
    return NULL;
  }
  
  PNOTICE("%s allocated %lu at %p", __func__, nsize, ptr);
  return PyMemoryView_FromMemory(ptr, nsize, PyBUF_WRITE);
}



/** 
 * Free direct memory allocated with allocate_direct_memory
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return 
 */
static PyObject * pymmcore_free_direct_memory(PyObject * self,
                                              PyObject * args,
                                              PyObject * kwds)
{
  static const char *kwlist[] = {"memory",
                                 NULL};

  PyObject * memview = NULL;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "O",
                                    const_cast<char**>(kwlist),
                                    &memview)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (PyMemoryView_Check(memview) == 0) {
    PyErr_SetString(PyExc_RuntimeError,"argument should be memoryview type");
    return NULL;
  }
  
  Py_buffer * buffer = PyMemoryView_GET_BUFFER(memview);
  buffer->len = 0;
  PyBuffer_Release(buffer);
  PNOTICE("%s freed memory %p", __func__, buffer->buf);
  Py_RETURN_NONE;
}

static PyObject * pymmcore_version(PyObject * self,
                               PyObject * args,
                               PyObject * kwds)
{
  return PyUnicode_FromString(PYMMCORE_API_VERSION);
}
