#include <common/queue.h>

namespace Common {
// Global static pointer used to ensure a single instance of the class.
Thread_ipc *Thread_ipc::_ipc = nullptr;

/** This function is called to create an instance of the class.
    Calling the constructor publicly is not allowed. The constructor
    is private and is only called by this instance function.
*/

Thread_ipc::Thread_ipc()
    : queue_to_mgr(QUEUE_SIZE,
                   Mpmc_bounded_lfq<struct message *>::allocate_queue_memory(
                       QUEUE_SIZE)),
      queue_to_ado(QUEUE_SIZE,
                   Mpmc_bounded_lfq<struct message *>::allocate_queue_memory(
                       QUEUE_SIZE)) {}

Thread_ipc *Thread_ipc::instance() {
  if (!_ipc) { // Only allow one instance of class to be generated.
    _ipc = new Thread_ipc;
  }
  return _ipc;
}

bool Thread_ipc::schedule_to_mgr(unsigned int shard_core, std::string &cores,
                                 float core_number, numa_node_t numa_zone) {
  message *msg = new message;
  msg->shard_core = shard_core;
  msg->op = Operation::schedule;
  msg->cores = cores;
  msg->core_number = core_number;
  msg->numa_zone = numa_zone;
  return queue_to_mgr.enqueue(msg);
}

bool Thread_ipc::register_to_mgr(unsigned int shard_core, std::string &cores,
                                 std::string id) {
  message *msg = new message;
  msg->shard_core = shard_core;
  msg->op = Operation::register_ado;
  msg->cores = cores;
  msg->ado_id = id;
  return queue_to_mgr.enqueue(msg);
}

bool Thread_ipc::schedule_to_ado(unsigned int shard_core,
                                 const std::string &cores, float core_number,
                                 numa_node_t numa_zone) {
  message *msg = new message;
  msg->shard_core = shard_core;
  msg->op = Operation::schedule;
  msg->cores = cores;
  msg->core_number = core_number;
  msg->numa_zone = numa_zone;
  return queue_to_ado.enqueue(msg);
}

bool Thread_ipc::kill_to_ado(unsigned int shard_core) {
  message *msg = new message;
  msg->shard_core = shard_core;
  msg->op = Operation::kill;
  return queue_to_ado.enqueue(msg);
}

bool Thread_ipc::get_next_ado(struct message *&msg) {
  return queue_to_ado.dequeue(msg);
}

bool Thread_ipc::get_next_mgr(struct message *&msg) {
  return queue_to_mgr.dequeue(msg);
}

bool Thread_ipc::add_to_ado(struct message *msg) {
  return queue_to_ado.enqueue(msg);
}

bool Thread_ipc::add_to_mgr(struct message *msg) {
  return queue_to_mgr.enqueue(msg);
}

void Thread_ipc::cleanup() {
  queue_to_ado.exit_threads();
  queue_to_mgr.exit_threads();
  struct message *msg = NULL;
  while (queue_to_ado.dequeue(msg)) {
    delete msg;
  }

  while (queue_to_mgr.dequeue(msg)) {
    delete msg;
  }

  delete msg;

  delete this;
}
} // namespace Common