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

#ifndef __ADOPROTO_H__
#define __ADOPROTO_H__

#include <common/errors.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/types.h>
#include <common/utils.h>
#include <unistd.h>
#include <cassert>
#include <cstddef>
#include <string>

#include "uipc.h"

class Buffer_header;

/**
 * Class to help UIPC message construction (both sides)
 *
 */
class ADO_protocol_builder
{
public:
  static constexpr size_t MAX_MESSAGE_SIZE  = 4096; 
  static constexpr size_t QUEUE_SIZE        = 8;
  static constexpr size_t POLL_SLEEP_USEC   = 500000;
  static constexpr size_t POLL_RETRY_LIMIT  = 1000000;
private:
  ADO_protocol_builder(const ADO_protocol_builder &) = delete;
  ADO_protocol_builder& operator=(const ADO_protocol_builder &) = delete;
  Buffer_header *msg_build_chirp_hello();
  
public:
  enum class Role {
    CONNECT,
    ACCEPT,
  };

  ADO_protocol_builder(const std::string& channel_prefix,
                       Role role);

  ~ADO_protocol_builder();

  void create_uipc_channels();

  status_t recv_bootstrap_response();

  void send_bootstrap();

  void send_bootstrap_response();

  void send_shutdown();

  /* shard-side, must not block */
  void send_memory_map(uint64_t token,
                       size_t size,
                       const void * value_vaddr);

  /* shard-side, must not block */
  void send_work_request(const uint64_t work_key,
                         const std::string& work_key_str,
                         const void * value_vaddr,
                         const size_t value_len,
                         const void * invocation_data,
                         const size_t invocation_data_len);

  void send_work_response(status_t status,
                          uint64_t work_key,
                          const void * response_data,
                          const size_t response_data_len);

  ssize_t recv_from_proxy(void * target, const size_t target_len);

  /* shard-side, must not block */
  bool recv_from_ado_work_completion(uint64_t& work_key,
                                     status_t& status,
                                     void*& response,
                                     size_t& response_len);

  /* table operations */
  void send_table_op_create(const uint64_t work_request_id,
                            const std::string& key,
                            const size_t value_len);

  void send_table_op_open(const uint64_t work_request_id,
                          const std::string& key);

  void send_table_op_resize(const uint64_t work_request_id,
                            const std::string& key,
                            const size_t new_value_len);

  void send_table_op_erase(const uint64_t work_request_id,
                           const std::string& key);

  void send_table_op_allocate_pool_memory(const uint64_t work_request_id,
                                          const size_t size,
                                          const size_t alignment);

  void send_table_op_free_pool_memory(const uint64_t work_request_id,
                                      const void * ptr,
                                      const size_t size);



  /* shard-side, must not block */
  bool recv_table_op_request(uint64_t& work_request_id,
                             int& op,
                             std::string& key,
                             size_t& value_len,
                             size_t& value_alignment,
                             void*& addr);


  void send_table_op_response(const status_t status,
                              const void * value_addr = nullptr,
                              size_t value_len = 0);

  void recv_table_op_response(status_t& status,
                              void *& out_value_addr,
                              size_t * p_out_value_len = nullptr);

  /* UIPC helpers */
  inline void * allocate_buffer() { return uipc_alloc_message(_channel); }
  inline void free_buffer(void * p) { assert(p); uipc_free_message(_channel, p); }
  inline status_t send(void * buffer) { return uipc_send(_channel, buffer); }
  inline status_t send_callback(void * buffer) { return uipc_send(_channel_callback, buffer); }
  inline status_t recv(channel_t ch, Buffer_header *& out_buffer) { void *v; auto rc = uipc_recv(ch, &v); out_buffer = static_cast<Buffer_header *>(v); return rc; }
  inline status_t recv(Buffer_header *& out_buffer) { return recv(_channel, out_buffer); }
  inline status_t recv_callback(Buffer_header *& out_buffer) { return recv(_channel_callback, out_buffer); }
  inline size_t max_message_size() const { return MAX_MESSAGE_SIZE; }
  /* out of line, to avoid exposing Buffer_header layout */
  static const uint8_t * buffer_header_to_message(Buffer_header *buffer);

  status_t poll_recv(Buffer_header *& out_buffer) {
    status_t s;
    while((s = recv(out_buffer)) == E_EMPTY) {
      cpu_relax();
    }
    return s;
  }

  status_t poll_recv_sleep(Buffer_header *& out_buffer) {
    status_t s;
    unsigned retries = 0;
    while((s = recv(out_buffer)) == E_EMPTY) {
      if(retries++ < POLL_RETRY_LIMIT)
        cpu_relax();
      else
        usleep(POLL_SLEEP_USEC);
    }
    return s;
  }

  status_t poll_recv_callback(Buffer_header *& out_buffer) {
    status_t s;
    while((s = recv_callback(out_buffer)) == E_EMPTY) {
      cpu_relax();
    }

    return s;
  }

private:
  std::string _channel_prefix;
  channel_t   _channel = nullptr;
  channel_t   _channel_callback = nullptr;
  
};


#endif
