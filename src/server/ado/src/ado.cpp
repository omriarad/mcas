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

#include <common/logging.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <api/interfaces.h>
#include <nupm/mcas_mod.h>
#include <atomic>
#include <boost/program_options.hpp>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <queue>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "ado.h"
#include "ado_proto.h"
#include "ado_proto_generated.h"


using namespace std;

// helpers

// globals
Component::IADO_plugin * i_plugin = nullptr;
ADO_protocol_builder * ipc = nullptr;

/* Callback functions */

static status_t ipc_create_key(uint64_t work_request_id,
                                    const std::string& key_name,
                                    const size_t value_size,
                                    void*& out_value_addr)
{
  status_t rc;
  ipc->send_table_op_create(work_request_id, key_name, value_size);
  ipc->recv_table_op_response(rc, out_value_addr);
  return rc;
}

static status_t ipc_open_key(const uint64_t work_request_id,
                                  const std::string& key_name,
                                  void*& out_value_addr,
                                  size_t& out_value_len)
{
  status_t rc;
  ipc->send_table_op_open(work_request_id, key_name);
  ipc->recv_table_op_response(rc, out_value_addr, &out_value_len);
  return rc;
}

static status_t ipc_erase_key(const uint64_t work_request_id,
                              const std::string& key_name)
{
  status_t rc;
  void* na;
  ipc->send_table_op_erase(work_request_id, key_name); 
  ipc->recv_table_op_response(rc, na);
  return rc;
}

static status_t ipc_resize_value(const uint64_t work_request_id,
                                 const std::string& key_name,
                                 const size_t new_value_size,
                                 void*& out_new_value_addr)
{
  status_t rc;
  ipc->send_table_op_resize(work_request_id, key_name, new_value_size);
  ipc->recv_table_op_response(rc, out_new_value_addr);
  return rc;
}


static status_t ipc_allocate_pool_memory(const uint64_t work_id,
                                         const size_t size,
                                         const size_t alignment,
                                         void *&out_new_addr)
{
  status_t rc;
  ipc->send_table_op_allocate_pool_memory(work_id, size, alignment);
  ipc->recv_table_op_response(rc, out_new_addr);
  return rc;
}

static status_t ipc_free_pool_memory(const uint64_t work_id,
                                     const size_t size,
                                     void * addr)
{
  status_t rc;
  void * na;
  ipc->send_table_op_free_pool_memory(work_id, addr, size);
  ipc->recv_table_op_response(rc, na);
  return rc;  
}



/** 
 * Main entry point
 * 
 * @param argc 
 * @param argv 
 * 
 * @return 
 */
int main(int argc, char* argv[])
{
  std::string plugin, channel_id;
  unsigned debug_level;
  std::string cpu_mask;

  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help", "Print help")
      ("plugin", po::value<std::string>(&plugin)->required(), "ADO plugin")
      ("channel_id", po::value<std::string>(&channel_id)->required(), "Channel (prefix) identifier")
      ("debug", po::value<unsigned>(&debug_level)->default_value(0), "Debug level")
      ("cpumask", po::value<std::string>(&cpu_mask), "Cores to restrict threads to; in string form")
      ;

    po::variables_map vm;
    try {
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help")) {
        cout << desc << endl;
        return 0;
      }
      po::notify(vm);
    }
    catch (po::error &e) {
      cerr << e.what() << endl;
      cerr << desc << endl;
      return -1;
    }
  }
  catch (exception &e) {
    cerr << e.what() << endl;
    return -1;
  }

  /* load plugin and register callbacks */
  i_plugin = static_cast<Component::IADO_plugin*>(load_component(plugin.c_str(), Interface::ado_plugin));
  if(!i_plugin)
    throw General_exception("unable to load ADO plugin (%s)", plugin.c_str());

  i_plugin->register_callbacks(Component::IADO_plugin::
                               Callback_table{ipc_create_key, ipc_open_key, ipc_erase_key, ipc_resize_value, ipc_allocate_pool_memory, ipc_free_pool_memory});
  PLOG("ADO: plugin loaded OK! (%s)", plugin.c_str());

  /* main loop */
  using namespace ADO_protocol;
  using namespace flatbuffers;

  unsigned long count = 0;

  ipc = new ADO_protocol_builder(channel_id, ADO_protocol_builder::Role::ACCEPT);
  bool exit = false;

  if(cpu_mask.empty() == false) {    
    cpu_mask_t m;
    if(string_to_mask(cpu_mask, m) == S_OK) {
      if(set_cpu_affinity_mask(m) == -1)
        throw Logic_exception("bad mask parameter");
      PLOG("ADO process configured with cpu mask: %s", cpu_mask.c_str());
    }
  }
    
  PLOG("ADO process: main thread (%p)", pthread_self());

  while (!exit) {

    /* main loop servicing incoming IPC requests */

    if(debug_level > 2)
      PLOG("ADO process: waiting for message (%lu)", count);

    Buffer_header * buffer = nullptr; /* recv will dequeue this */
    
    /* poll until there is a request, sleep on too much polling  */
    ipc->poll_recv_sleep(buffer);
    assert(buffer);

    if(debug_level > 2)
      PMAJOR("ADO: got new IPC message");

    auto protocol_start = ipc->buffer_header_to_message(buffer);
    auto msg = GetMessage(protocol_start);

    if(msg) {

      /* SHARD->ADO : incoming from shard process */
      auto element_type = msg->element_type();
      switch(element_type)
        {
        case Element_Chirp:
          {
            auto chirp = msg->element_as_Chirp();
            auto type = chirp->chirp_type();
            switch(type)
              {
              case ChirpType_Hello:
                PMAJOR("ADO: received Hello chirp");
                ipc->send_bootstrap_response();
                break;
              case ChirpType_Shutdown:
                PMAJOR("ADO: received Shutdown chirp");
                exit = true;
                break;
              default:
                throw Protocol_exception("unknown chirp");
            }
            break;
          }
        case Element_MapMemory:
          {
            auto mm = msg->element_as_MapMemory();
            assert(mm);

            size_t mm_size = 0;
            void * mm_addr = nullptr;
            /* use same of shard virtual address for the moment */
            if(!nupm::check_mcas_kernel_module())
              throw General_exception("inaccessible MCAS kernel module");

            if((mm_addr = nupm::mmap_exposed_memory(mm->token(), mm_size, (void*) mm->shard_addr())) == (void*) -1)
              throw General_exception("nupm::mmap_exposed_memory: failed unexpectedly");

            //            touch_pages(mm_addr, mm_size);
            PMAJOR("ADO: mapped memory %lx size:%lu", mm->token(), mm->size());

            /* register with plugin */
            if(i_plugin->register_mapped_memory((void*)mm->shard_addr(), mm_addr, mm_size) != S_OK)
              throw General_exception("calling register_mapped_memory on ADO plugin failed");

            break;
          }
        case Element_WorkRequest:
          {
            auto wr = msg->element_as_WorkRequest();
            void * out_work_response = nullptr;
            size_t out_work_response_len = 0;

            /* forward to plugin */
            status_t rc = i_plugin->do_work(wr->work_key(),
                                            wr->work_key_string()->c_str(),
                                            (void*) wr->value_addr(),
                                            wr->value_len(),
                                            wr->request()->Data(),
                                            wr->request()->Length(),
                                            out_work_response,
                                            out_work_response_len);

            /* pass back response data */
            ipc->send_work_response(rc,
                                    wr->work_key(),
                                    out_work_response,
                                    out_work_response_len);

            ::free(out_work_response);
            break;
          }
        default:
          {
            PLOG("ADO: unknown element type: %d", element_type);
          }
        }
    }
    else throw Protocol_exception("BAD ADO message");


    ipc->free_buffer(buffer);
    count++;
  }

  /* any cleanup */
}




// (const std::string& key,
//                                   void * shard_value_vaddr,
//                                   size_t value_len,
//                                   const std::vector<uint8_t>& work_request,
//                                   std::vector<uint8_t>& work_response)
