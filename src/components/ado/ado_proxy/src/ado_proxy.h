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

#ifndef __ADOPROX_COMPONENT_H__
#define __ADOPROX_COMPONENT_H__

#include "ado_proto.h"
#include "docker.h"
#include <api/ado_itf.h>
#include <api/kvstore_itf.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>

#define PROX_TO_ADO 1
#define RECV_EXIT 2
#define RECV_COMPLETE 3

using namespace std;
using namespace Component;

class ADO_proxy : public IADO_proxy {
public:
  ADO_proxy(Component::IKVStore::pool_t pool_id, const std::string &filename,
            std::vector<std::string> &args, std::string cores, int memory,
            float core_number, numa_node_t numa_zone);

  virtual ~ADO_proxy();

  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0x8a120985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75,
                         0x21, 0xa1, 0x92); //

  void *query_interface(Component::uuid_t &itf_uuid) override {
    if (itf_uuid == Component::IADO_proxy::iid()) {
      return (void *)static_cast<Component::IADO_proxy *>(this);
    } else
      return NULL; // we don't support this interface
  }

  void unload() override {
    kill();
    delete this;
  }

  status_t bootstrap_ado() override;

  status_t send_memory_map(uint64_t token, size_t size,
                           void *value_vaddr) override;

  status_t send_work_request(const uint64_t work_request_key,
                             const std::string &work_key_str,
                             const void *value_addr, const size_t value_len,
                             const void *invocation_data,
                             const size_t invocation_data_len) override;


  bool check_work_completions(uint64_t& request_key,
                              status_t& out_status,
                              void *& out_response, /* use ::free to release */
                              size_t & out_response_length) override;

  bool check_table_ops(uint64_t &work_request_id,
                       int &op,
                       std::string &key,
                       size_t &value_len,
                       size_t &value_alignment,
                       void *& addr) override;

  void send_table_op_response(const status_t s, const void *value_addr,
                              size_t value_len) override;

  bool has_exited() override;

  status_t shutdown() override;

  void add_deferred_unlock(const uint64_t work_key,
                           const Component::IKVStore::key_t key) override;

  void
  get_deferred_unlocks(const uint64_t work_key,
                       std::vector<Component::IKVStore::key_t> &keys) override;

  std::string ado_id() const { return container_id; }

  Component::IKVStore::pool_t pool_id() const { return _pool_id; }

private:
  status_t kill();
  void launch();

  std::unique_ptr<ADO_protocol_builder> _ipc;
  std::string _cores;
  float _core_number;
  std::string _filename;
  vector<string> _args;
  Component::IKVStore::pool_t _pool_id;
  std::string _channel_name;
  int _memory;
  numa_node_t _numa;
  std::map<uint64_t, std::vector<Component::IKVStore::key_t>> _deferred_unlocks;
  unsigned _outstanding_wr = 0;

  /* these need renaming */
  work_id_t id;
  string container_id;
  pid_t pid;
  key_t key;
  DOCKER *docker;
};

class ADO_proxy_factory : public Component::IADO_proxy_factory {
public:
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0xfac20985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75,
                         0x21, 0xa1, 0x92); //

  void *query_interface(Component::uuid_t &itf_uuid) override {
    if (itf_uuid == Component::IADO_proxy_factory::iid()) {
      return (void *)static_cast<Component::IADO_proxy_factory *>(this);
    } else
      return NULL; // we don't support this interface
  }

  void unload() override { delete this; }

  virtual Component::IADO_proxy *
  create(Component::IKVStore::pool_t pool_id, const std::string &filename,
         std::vector<std::string> &args, std::string cores, int memory,
         float cpu_num, numa_node_t numa_zone) override {
    Component::IADO_proxy *obj =
        static_cast<Component::IADO_proxy *>(new ADO_proxy(
            pool_id, filename, args, cores, memory, cpu_num, numa_zone));
    assert(obj);
    obj->add_ref();
    return obj;
  }
};

#endif
