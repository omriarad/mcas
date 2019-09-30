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
#include <libpmem.h>
#include <common/logging.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/cycles.h>
#include <api/interfaces.h>
#include <iostream>
#include <string.h>
#include <ccpm/value.h>
#include <ccpm/record.h>
#include <ccpm/immutable_string_table.h>
#include "graph_plugin.h"

//#include <flatbuffers/flatbuffers.h>

status_t ADO_graph_plugin::register_mapped_memory(void * shard_vaddr,
                                                 void * local_vaddr,
                                                 size_t len)
{
  PLOG("ADO_graph_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr, local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */

  return S_OK;
}

#include <libpmem.h>




status_t ADO_graph_plugin::do_work(uint64_t work_key,
                                   const std::string& key,
                                   void * value,
                                   size_t value_len,
                                   const void * in_work_request, /* don't use iovec because of non-const */
                                   const size_t in_work_request_len,
                                   void*& out_work_response,
                                   size_t& out_work_response_len)
{
  using namespace ccpm;
  using namespace Graph_ADO_protocol;
  using namespace flatbuffers;
  
  PLOG("key:%s value:%p value_len:%lu", key.c_str(), value, value_len);
  PLOG("work_request: %p len=%lu", in_work_request, in_work_request_len);

  if(key == "transaction")  {
    Verifier verifier(static_cast<const uint8_t*>(in_work_request), in_work_request_len);
  
    if(!VerifyMessageBuffer(verifier)) {
      throw General_exception("data verify failed");
      return E_FAIL;
    }
    
    auto msg = GetMessage(in_work_request);
    auto transaction = msg->element_as_Transaction();
    if(transaction)
      return handle_transaction(transaction, work_key, value, value_len);    
  }

  return S_OK;
}

status_t ADO_graph_plugin::handle_transaction(const Graph_ADO_protocol::Transaction * transaction,
                                              uint64_t work_key,
                                              void * value,
                                              size_t value_len)
{
  using namespace ccpm;

  static bool first = true;
  PLOG("transaction: %s %s", transaction->target()->c_str(), transaction->source()->c_str());
  assert(value_len >= sizeof(Cow_value_pointer<int>));

  Cow_value_pointer<int> cvp(value, value_len, TYPE_POINTER, first);
  if(first || cvp.get_current_version()->ptr == nullptr) {
    PLOG("rebuilt:");
    Cow_value_pointer<int>::Persistent_version * pv;
    unsigned slot = cvp.get_new_version_slot(pv);
    pv->ptr = new (allocate_memory(work_key, 64, 8)) int;
    pv->length = sizeof(int);
    cvp.atomic_commit_version(slot);
    first = false;
  }
  else {
    PLOG("existing:");
  }
  cvp.dump_info();
  
  asm("int3");
  return S_OK;
}
                                  
status_t ADO_graph_plugin::shutdown()
{
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/** 
 * Factory-less entry point 
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& interface_iid)
{
  PLOG("instantiating ADO_graph_plugin");
  if(interface_iid == Interface::ado_plugin) 
    return static_cast<void*>(new ADO_graph_plugin());
  else return NULL;
}



  
  // if(VerifyPropertySetBuffer(verifier)==false)
  //   throw General_exception("bad data");
  
  //  Versioned_record_type record(shard_value_vaddr, value_len);
  
  // ccpm::Immutable_string_table<> * symtab;
  
  // if(strncmp("st-init",(const char *)in_work_request,7)==0) {
  //   symtab = new ccpm::Immutable_string_table<>(shard_value_vaddr, value_len);
  //   return S_OK;
  // }

  // return S_OK;
  // symtab = new ccpm::Immutable_string_table<>(shard_value_vaddr);
  //  auto new_str_ptr = symtab->add_string((const char *) in_work_request,  in_work_request_len);
  //  PLOG("added string: %p (%s)", new_str_ptr, (const char *) new_str_ptr);
  //  
  //  PLOG("ADO_graph_plugin: work_id (%lu)", work_key);
  //  PLOG("ADO_graph_plugin: do_work (%s, value_addr=%p, valuen_len=%lu)", key.c_str(), shard_value_vaddr, value_len);
  // PLOG("Current value: %.*s", (int) value_len, (char *) shard_value_vaddr); 
  // memset(shard_value_vaddr, 'X', value_len);  
  // nupm::mem_flush(shard_value_vaddr, value_len);
  // out_work_response_len = 3;
  // out_work_response = ::malloc(out_work_response_len);
  // strncpy((char*) out_work_response, "OK!", 3);
  
  // /* test callback */
  // void * new_value_addr = nullptr;
  // _func_create_key(work_key, "newKey", 12, new_value_addr);
  // memset(new_value_addr,'N', 12);
  // PLOG("new key created at %p", new_value_addr);

