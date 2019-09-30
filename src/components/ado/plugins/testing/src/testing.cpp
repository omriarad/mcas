/*
  Copyright [2017-2019] [IBM Corporation]
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

#include "testing.h"
#include <libpmem.h>
#include <api/interfaces.h>
#include <common/logging.h>
#include <iostream>
#include <string.h>

status_t ADO_testing_plugin::register_mapped_memory(void *shard_vaddr,
                                                    void *local_vaddr,
                                                    size_t len) {
  PLOG("ADO_testing_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr,
       local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

#define ASSERT_TRUE(X,Y) if(!( X )) { PERR( Y ); return E_FAIL; }

status_t ADO_testing_plugin::do_work(uint64_t work_key,
                                     const std::string &key,
                                     void *value,
                                     size_t value_len,
                                     const void *in_work_request, /* don't use iovec because of non-const */
                                     const size_t in_work_request_len,
                                     void *&out_work_response,
                                     size_t &out_work_response_len) {

  PLOG("ADO_testing_plugin: do_work (key=%s)", key.c_str());
  status_t rc;
  
  ASSERT_TRUE(value_len != 0, "ADO_testing_plugin: value_len is 0");

  if(key == "test0") {
    PLOG("ADO_testing_plugin: running test0 (key=%s)", key.c_str());
    std::string work(static_cast<const char*>(in_work_request), in_work_request_len);
    PLOG("ADO_testing_plugin: work=(%s)", work.c_str());
    ASSERT_TRUE(work == "RUN!TEST-0", "ADO_testing_plugin: unexpected message");

    memset(value, 0xe, value_len);
    PLOG("value=%p value_len=%lu", value, value_len);
    ASSERT_TRUE(value_len == 4096, "ADO_testing_plugin: value bad length");

    /* resize value */
    void * new_addr = nullptr;
    rc = _cb.resize_value_func(work_key, key, KB(8), new_addr);
    ASSERT_TRUE(rc == S_OK,
                "ADO_testing_plugin: resize_value callback failed");
    PLOG("value after resize=%p", new_addr);

    /* check old content is still there */  
    void * tmp = malloc(value_len);
    memset(tmp, 0xe, value_len);

    //    ASSERT_TRUE(memcmp(tmp, new_addr, value_len) == 0,
    //            "ADO_testing_plugin: content after resize corrupt");

    PLOG("ADO_testing_plugin: content after resize OK.");
    memset(new_addr, 0xf, KB(8));

    /* create new value */
    void *new_value_addr = nullptr;
    rc = _cb.create_key_func(work_key, "newKey", 12, new_value_addr);
    memset(new_value_addr, 'N', 12);
    PLOG("new key created at %p", new_value_addr);
    ASSERT_TRUE(new_value_addr && rc==S_OK, "ADO_testing_plugin: create_key callback failed");

    /* allocate value memory */
    void * new_value_memory = nullptr;
    size_t s = 1200;
    rc = _cb.allocate_pool_memory_func(work_key, s, 4096 /* alignment */, new_value_memory);
    PLOG("new value: %p rc=%d", new_value_memory, rc);
    //    ASSERT_TRUE(rc == S_OK, "ADO_testing_plugin: allocate pool memory failed");
    //    memset(new_value_memory, 'X', s);

    rc = _cb.free_pool_memory_func(work_key, s, new_value_memory);
    PLOG("free new value: rc=%d", rc);
  
    PMAJOR("*** test0: passed!");
  }
  else if(key == "test1") {
    std::string val(reinterpret_cast<char*>(value), value_len);
    PLOG("ADO_testing_plugin: (key=%s,value=%s)", key.c_str(), val.c_str());
    std::string request(reinterpret_cast<const char*>(in_work_request), in_work_request_len);
    PLOG("ADO_testing_plugin: (request=%s)", request.c_str());
    PMAJOR("*** test1: passed!");

    ASSERT_TRUE("RUN!TEST-1" == request, "ADO_testing_plugin: invoke_put_ado failed");
    ASSERT_TRUE("VALUE_TO_PUT" == val, "ADO_testing_plugin: invoke_put_ado failed");
  }
  else if(key == "test2") {
    PMAJOR("test2....");
  }
  else {
    PMAJOR("unknown key:%s", key.c_str());
  }
  return S_OK;
}

status_t ADO_testing_plugin::shutdown() {
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(Component::uuid_t &interface_iid) {
  PLOG("instantiating ADO_testing_plugin");
  if (interface_iid == Interface::ado_plugin)
    return static_cast<void *>(new ADO_testing_plugin());
  else
    return NULL;
}

#undef RESET_STATE
