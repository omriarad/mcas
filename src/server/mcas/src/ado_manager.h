
#ifndef __ADO_MANAGER_H__
#define __ADO_MANAGER_H__

#include "config_file.h"
#include "program_options.h"
#include <common/queue.h>
#include <common/types.h>
#include <set>
#include <thread>
#include <utility>
#include <vector>

class SLA;

struct ado {
  unsigned int shard_id;
  std::string cpus;
  // can be container id
  std::string id;
};

struct compare {
  bool operator()(const std::pair<unsigned int, unsigned int> &lhs,
                  const std::pair<unsigned int, unsigned int> &rhs) const {
    if (lhs.second == rhs.second)
      return (lhs.first < rhs.first);
    return (lhs.second < rhs.second);
  }
};

class ADO_manager : public mcas::Config_file {
public:
  ADO_manager(Program_options &options)
      : mcas::Config_file(options.config_file),
        _thread(&ADO_manager::init, this) {
    _sla = NULL;
  }
  ~ADO_manager() { exit(); }
  void setSLA();

private:
  std::thread _thread;
  SLA *_sla = NULL;
  std::vector<struct ado> _ados;
  // std::set<std::pair<unsigned int, unsigned int>, compare> _ado_cpu_pool;
  std::set<unsigned int> _ado_cpu_pool;
  std::set<unsigned int> _manager_cpu_pool;
  bool _exit = false;

  void init();
  void main_loop();
  inline void exit() {
    _exit = true;
    for (auto &ado : _ados) {
      kill_ado(ado);
    }

    Common::Thread_ipc::instance()->cleanup();
    _thread.join();
  }
  void resource_check() {
    // TODO
  }
  void register_ado(unsigned int, std::string &, std::string &);
  void kill_ado(const struct ado &ado);
  void schedule(unsigned int shard_core, std::string cpus, float cpu_num,
                numa_node_t numa_zone);
  std::string reschedule(const struct ado &ado) {
    // TODO
    return ado.cpus;
  }
};

#endif
