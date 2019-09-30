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



/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */
#include "uipc_channel.h"

#include "uipc_shared_memory.h"
#include <common/errors.h>
#include <common/exceptions.h>

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

#include <unistd.h>
#include <cassert>
#include <string>

namespace Core
{
namespace UIPC
{
// typedef Common::Spsc_bounded_lfq_sleeping<Dataplane::Command_t, 32 /* queue
// size */> command_queue_t;

Channel::Channel(const std::string &name, size_t message_size, size_t queue_size)
    : _master(true)
    , _shmem_fifo_m2s()
    , _shmem_fifo_s2m()
    , _shmem_slab_ring()
    , _shmem_slab() {
  size_t queue_footprint = queue_t::memory_footprint(queue_size);
  size_t pages_per_queue = round_up(queue_footprint, PAGE_SIZE) / PAGE_SIZE;

  assert((queue_size != 0) && ((queue_size & (~queue_size + 1)) ==
                               queue_size));  // queue len is a power of 2
  assert(message_size % 8 == 0);

  if(option_DEBUG)
    PLOG("pages per FIFO queue: %zu", pages_per_queue);

  size_t slab_queue_pages =
      round_up(queue_t::memory_footprint(queue_size * 2), PAGE_SIZE) / PAGE_SIZE;

  if(option_DEBUG)
    PLOG("slab_queue_pages: %ld", slab_queue_pages);

  size_t slab_pages =
      round_up(message_size * queue_size * 2, PAGE_SIZE) / PAGE_SIZE;

  if(option_DEBUG)
    PLOG("slab_pages: %ld", slab_pages);

  size_t total_pages = ((pages_per_queue * 2) + slab_queue_pages + slab_pages);

  if(option_DEBUG)
    PLOG("total_pages: %ld", total_pages);

  _shmem_fifo_m2s = new Shared_memory(name + "-m2s", pages_per_queue);
  _shmem_fifo_s2m = new Shared_memory(name + "-s2m", pages_per_queue);
  _shmem_slab_ring = new Shared_memory(name + "-slabring", slab_queue_pages);
  _shmem_slab = new Shared_memory(name + "-slab", slab_pages);

  _out_queue = new (_shmem_fifo_m2s->get_addr()) queue_t(
      queue_size, (static_cast<char*>(_shmem_fifo_m2s->get_addr())) + sizeof(queue_t));

  _in_queue = new (_shmem_fifo_s2m->get_addr()) queue_t(
      queue_size, (static_cast<char*>(_shmem_fifo_s2m->get_addr())) + sizeof(queue_t));

  size_t slab_slots = queue_size * 2;
  _slab_ring = new (_shmem_slab_ring->get_addr()) queue_t(
      slab_slots, (static_cast<char*>(_shmem_slab_ring->get_addr())) + sizeof(queue_t));
  byte* slot_addr = static_cast<byte*>(_shmem_slab->get_addr());
  for (size_t i = 0; i < slab_slots; i++) {
    _slab_ring->enqueue(slot_addr);
    slot_addr += message_size;
  }
}

Channel::Channel(const std::string &name) : _master(false)
  , _shmem_fifo_m2s(new Shared_memory(name + "-m2s"))
  , _shmem_fifo_s2m(new Shared_memory(name + "-s2m"))
  , _shmem_slab_ring(new Shared_memory(name + "-slabring"))
  , _shmem_slab(new Shared_memory(name + "-slab")) {

  if(option_DEBUG) {
    PMAJOR("got fifo (m2s) @ %p - %lu bytes", _shmem_fifo_m2s->get_addr(),
           _shmem_fifo_m2s->get_size());

    PMAJOR("got fifo (s2m) @ %p - %lu bytes", _shmem_fifo_s2m->get_addr(),
           _shmem_fifo_s2m->get_size());

    PMAJOR("got slab ring @ %p - %lu bytes", _shmem_slab_ring->get_addr(),
           _shmem_slab_ring->get_size());

    PMAJOR("got slab @ %p - %lu bytes", _shmem_slab->get_addr(),
           _shmem_slab->get_size());
  }

  _in_queue = reinterpret_cast<queue_t*>(_shmem_fifo_m2s->get_addr());
  _out_queue = reinterpret_cast<queue_t*>(_shmem_fifo_s2m->get_addr());
  _slab_ring = reinterpret_cast<queue_t*>(_shmem_slab_ring->get_addr());

  ::usleep(500000); /* hack to let master get ready - could improve with state in
                     shared memory */
}

Channel::~Channel() {
  /* don't delete queues since they were constructed on shared memory */

  delete _shmem_fifo_m2s;
  delete _shmem_fifo_s2m;
  delete _shmem_slab_ring;
  delete _shmem_slab;
}

status_t Channel::send(void* msg) {
  assert(_out_queue);
  if (_out_queue->enqueue(msg)) {
    return S_OK;
  }
  else {
    return E_FULL;
  }
}

status_t Channel::recv(void*& recvd_msg) {
  assert(_in_queue);
  if (_in_queue->dequeue(recvd_msg))
    return S_OK;
  else
    return E_EMPTY;
}

void Channel::unblock_threads() { _in_queue->exit_threads(); }

void* Channel::alloc_msg() {
  assert(_slab_ring);
  void* msg = nullptr;
  _slab_ring->dequeue(msg);
  if (!msg) throw General_exception("channel: out of message slots");
  return msg;
}

status_t Channel::free_msg(void* msg) {
  assert(_slab_ring);
  assert(msg);
  /* currently no checks for validity */
  _slab_ring->enqueue(msg);
  return S_OK;
}

}  // namespace UIPC
}  // namespace Core
