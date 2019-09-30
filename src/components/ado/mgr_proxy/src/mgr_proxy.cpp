#include "mgr_proxy.h"
#include <api/components.h>
#include <assert.h>
#include <common/logging.h>
#include <fstream>
#include <iostream>
#include <sys/types.h>

using namespace Component;
using namespace std;
using namespace Common;

ADO_manager_proxy::ADO_manager_proxy(unsigned debug_level, int shard,
                                     std::string cores, float cpu_num)
    : debug_level(debug_level), shard(shard), cores(cores), cpu_num(cpu_num) {}

ADO_manager_proxy::ADO_manager_proxy() {
  debug_level = 0;
  shard = 0;
  cores = "";
  cpu_num = 1;
}

ADO_manager_proxy::~ADO_manager_proxy() {}

IADO_proxy *ADO_manager_proxy::create(Component::IKVStore::pool_t pool_id,
                                      const std::string &filename,
                                      std::vector<std::string> &args,
                                      numa_node_t value_memory_numa_zone,
                                      SLA *sla) {

  if (!Thread_ipc::instance()->schedule_to_mgr(shard, cores, cpu_num,
                                               value_memory_numa_zone)) {
    PERR("schedule_to_mgr thread ipc enqueue fail!");
  };

  struct message *msg = NULL;
  Thread_ipc::instance()->get_next_ado(msg);

  while (!msg || msg->shard_core != shard || msg->op != Operation::schedule) {
    if (msg)
      Thread_ipc::instance()->add_to_ado(msg);
    Thread_ipc::instance()->get_next_ado(msg);
  }

  // TODO:ask manager what memory;
  int memory = 10485760;

  Component::IBase *comp = Component::load_component(
      "libcomponent-adoproxy.so", Component::ado_proxy_factory);

  IADO_proxy_factory *fact = static_cast<IADO_proxy_factory *>(
      comp->query_interface(IADO_proxy_factory::iid()));

  assert(fact);

  IADO_proxy *ado = fact->create(pool_id, filename, args, msg->cores, memory,
                                 msg->core_number, msg->numa_zone);

  fact->release_ref();

  if (!Thread_ipc::instance()->register_to_mgr(shard, msg->cores,
                                               ado->ado_id())) {
    PERR("register_to_mgr enqueue failed!");
  };

  return ado;
}

bool ADO_manager_proxy::has_exited(IADO_proxy *ado) {
  return ado->has_exited();
}

status_t ADO_manager_proxy::shutdown(IADO_proxy *ado) {
  ado->release_ref();
  return 0;
}
/**
 * Factory entry point
 *
 */
extern "C" void *factory_createInstance(Component::uuid_t &component_id) {
  if (component_id == ADO_manager_proxy_factory::component_id()) {
    return static_cast<void *>(new ADO_manager_proxy_factory());
  } else
    return NULL;
}
