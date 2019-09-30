#ifndef __QUEUE_H_
#define __QUEUE_H_

#include "mpmc_bounded_queue.h"
#include "types.h"

#define QUEUE_SIZE 256

namespace Common {

enum class Operation { kill, schedule, reschedule, register_ado, wait };

struct message {
  unsigned int shard_core;
  Operation op;
  std::string cores;
  float core_number;
  std::string ado_id;
  numa_node_t numa_zone;
};

class Thread_ipc {
public:
  static Thread_ipc *instance();
  Thread_ipc(Thread_ipc const &) = delete; // copy constructor deleted
  Thread_ipc &operator=(const Thread_ipc &) = delete;
  bool schedule_to_mgr(unsigned int shard_core, std::string &cores,
                       float core_number, numa_node_t numa_zone);
  bool register_to_mgr(unsigned int shard_core, std::string &cores,
                       std::string id);
  bool schedule_to_ado(unsigned int shard_core, const std::string &cores,
                       float core_number, numa_node_t numa_zone);
  bool kill_to_ado(unsigned int shard_core);
  bool get_next_ado(struct message *&msg);
  bool get_next_mgr(struct message *&msg);
  bool add_to_ado(struct message *msg);
  bool add_to_mgr(struct message *msg);
  void cleanup();
  ~Thread_ipc() { cleanup(); }

private:
  Thread_ipc();
  static Thread_ipc *_ipc;
  Mpmc_bounded_lfq_sleeping<struct message *> queue_to_ado;
  Mpmc_bounded_lfq_sleeping<struct message *> queue_to_mgr;
};
} // namespace Common

#endif
