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

#include "ado_proto.h"
#include <common/logging.h>
#include <common/exceptions.h>
#include <common/dump_utils.h>
#include <boost/numeric/conversion/cast.hpp>
#include <cstring>
#include <cstddef>
#include <cstdlib>
#include <cinttypes> // PPIu32
#include <string>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Weffc++"
#include "flatbuffers/flatbuffers.h"
#include "ado_proto_generated.h"
#pragma GCC diagnostic push

using namespace flatbuffers;
using namespace ADO_protocol;

class Buffer_header
{
  /* size of the space returned by allocate_buffer()
   * not currently used, but would be handy if allocate_buffer
   * ever changed to return more than one size of buffer.
   */
  std::uint32_t _size;
  std::uint32_t _offset; // offset to the region used by flatbuffers
  const uint8_t *buf_begin() const { return static_cast<const uint8_t *>(static_cast<const void *>(this)); }
  uint8_t *buf_begin() { return static_cast<uint8_t *>(static_cast<void *>(this)); }
public:
  Buffer_header(std::uint32_t size_)
    : _size(size_)
    , _offset(0)
  {
  }
  const uint8_t * to_message() const
  {
    assert(_offset != 0);
    return buf_begin() + _offset;
  }
  uint8_t * to_message()
  {
    assert(_offset != 0);
    return buf_begin() + _offset;
  }
  const uint8_t *end() const
  {
    return buf_begin() + _size;
  }
  uint8_t *end()
  {
    return buf_begin() + _size;
  }
  void finish(const FlatBufferBuilder *fbb)
  {
    const uint8_t *msg_begin = fbb->GetBufferPointer();
    const uint8_t *msg_end = end();
    assert(static_cast<const uint8_t *>(static_cast<const void *>(this+1)) < msg_begin);
    assert(msg_begin <= msg_end);
    _offset = uint32_t(msg_begin - buf_begin());
    assert(_offset < ADO_protocol_builder::MAX_MESSAGE_SIZE);
  }
};

const uint8_t * ADO_protocol_builder::buffer_header_to_message(Buffer_header *bh)
{
  return bh->to_message();
}

ADO_protocol_builder::ADO_protocol_builder(const std::string& channel_prefix,
                                           Role role)
  : _channel_prefix(channel_prefix)
{
  /* connect UIPC channels */
  if(role == Role::CONNECT) {
  }
  else if(role == Role::ACCEPT) {
    _channel = uipc_connect_channel(_channel_prefix.c_str());
    _channel_callback = uipc_connect_channel(std::string(_channel_prefix + "-cb").c_str());
  }
  else throw Logic_exception("bad role");
}

ADO_protocol_builder::~ADO_protocol_builder()
{
  if(_channel)
    uipc_close_channel(_channel);
  if(_channel_callback)
    uipc_close_channel(_channel_callback);
}

void ADO_protocol_builder::create_uipc_channels()
{
  _channel = uipc_create_channel(_channel_prefix.c_str(),
                                 MAX_MESSAGE_SIZE,
                                 QUEUE_SIZE);

  _channel_callback = uipc_create_channel(std::string(_channel_prefix + "-cb").c_str(),
                                          MAX_MESSAGE_SIZE,
                                          QUEUE_SIZE);
}

class Protocol_allocator
  : public flatbuffers::Allocator
{
  ADO_protocol_builder *_builder;
  Buffer_header *_bh;
public:
  Protocol_allocator(ADO_protocol_builder *builder_)
    : flatbuffers::Allocator()
    , _builder(builder_)
    , _bh{}
  {}
  virtual uint8_t *allocate(size_t size)
  {
    if ( _bh )
    {
      throw std::domain_error("Protocol_allocator is good for only a single allocation.");
    }
	auto size32 = boost::numeric_cast<std::uint32_t>(size);
    if ( _builder->max_message_size() - sizeof(Buffer_header) < size32 )
    {
      throw std::domain_error("Protocol_allocator: size requested would not fit in a buffer.");
    }
    _bh = new (_builder->allocate_buffer()) Buffer_header{std::uint32_t(_builder->max_message_size())};
    /* Return only as much as flatbuffer expects.
     * This object contains the actual address.
     */
    return _bh->end() - size;
  }

  ~Protocol_allocator()
  {
    assert( std::current_exception() || ! _bh ); // except when processing an exception, _bh should have been released. Did you forget to do so?
  }

  uint8_t *reallocate_downward(uint8_t * // old_p
    , size_t // old_size
    , size_t // new_size
    , size_t // in_use_back
    , size_t // in_use_front
  ) override {
    throw std::range_error("Fixed size shared memory space. Cannot be reallocated.");
  }

  Buffer_header *release(const FlatBufferBuilder *fbb)
  {
    if ( ! _bh )
    {
      throw std::domain_error("release: No buffer available (never allocated or already released).");
    }
    _bh->finish(fbb);
    auto bh = _bh;
    _bh = nullptr;
    return bh;
  }

  void deallocate(
    uint8_t * // p
    , size_t // size
  )
  {
  }
};

class Flatbuffer_builder
  : public Protocol_allocator
  , private FlatBufferBuilder
{
public:
  Flatbuffer_builder(ADO_protocol_builder *apb)
    : Protocol_allocator(apb)
    , FlatBufferBuilder(apb->max_message_size() - sizeof(Buffer_header), static_cast<Protocol_allocator *>(this))
  {}
  FlatBufferBuilder &fbb() { return *this; }
  Buffer_header *release() {
    auto bh = static_cast<Protocol_allocator *>(this)->release(this);
    return bh;
  }
  friend class ADO_protocol_builder;
};

Buffer_header *ADO_protocol_builder::msg_build_chirp_hello()
{
  Flatbuffer_builder fbb(this);
  auto chirp = CreateChirp(fbb, ChirpType_Hello);
  auto msg = CreateMessage(fbb, Element_Chirp, chirp.Union());
  fbb.Finish(msg);

  return fbb.release();
}

void ADO_protocol_builder::send_bootstrap()
{
  Buffer_header * bh = msg_build_chirp_hello();

  send(bh);

  PLOG("%s", "ADO_protocol_builder: sent bootstrap");
}

void ADO_protocol_builder::send_bootstrap_response()
{
  Buffer_header * bh = msg_build_chirp_hello();

  send(bh);
}

void ADO_protocol_builder::send_shutdown()
{
  Flatbuffer_builder fbb(this);

  auto chirp = CreateChirp(fbb, ChirpType_Shutdown);
  auto msg = CreateMessage(fbb, Element_Chirp, chirp.Union());
  fbb.Finish(msg);

  send(fbb.release());
}


void ADO_protocol_builder::send_memory_map(uint64_t memory_token,
                                           size_t memory_size,
                                           const void * shard_address)
{
  Flatbuffer_builder fbb(this);

  auto mm = CreateMapMemory(fbb, memory_token, memory_size, (addr_t) shard_address);
  auto msg = CreateMessage(fbb, Element_MapMemory, mm.Union());
  fbb.Finish(msg);

  send(fbb.release());

  PLOG("%s", "ADO_protocol_builder::send_memory_map OK");
}


void ADO_protocol_builder::send_work_request(const uint64_t work_request_key,
                                             const std::string& work_key_str,
                                             const void * value_vaddr,
                                             const size_t value_len,
                                             const void * invocation_data,
                                             const size_t invocation_data_len)
{
  Flatbuffer_builder fbb(this);

  auto data = fbb.CreateVector(reinterpret_cast<const uint8_t*>(invocation_data), invocation_data_len);
  auto wks = fbb.CreateString(work_key_str);
  auto wr = CreateWorkRequest(fbb, work_request_key, wks, (addr_t) value_vaddr, value_len, data);
  auto msg = CreateMessage(fbb, Element_WorkRequest, wr.Union());
  fbb.Finish(msg);

  send(fbb.release());
}

void ADO_protocol_builder::send_work_response(status_t status,
                                              uint64_t work_key,
                                              const void * response_data,
                                              const size_t response_data_len)
{
  Flatbuffer_builder fbb(this);

  auto data = fbb.CreateVector(static_cast<const uint8_t*>(response_data), response_data_len);
  auto resp = CreateWorkResponse(fbb, work_key, status, data);
  auto msg = CreateMessage(fbb, Element_WorkResponse, resp.Union());
  fbb.Finish(msg);

  send(fbb.release());
}

bool ADO_protocol_builder::recv_from_ado_work_completion(uint64_t& work_key,
                                                         status_t& status,
                                                         void*& response,
                                                         size_t& response_len)
{
  Buffer_header * buffer = nullptr;
  status_t s = recv(buffer);

  if(s == E_EMPTY) {
    return false;
  }
  else if(s == S_OK) {
    
    assert(buffer);
    auto protocol_start = buffer->to_message();
    auto msg = GetMessage(protocol_start);
    auto wr = msg->element_as_WorkResponse();
    if(wr == nullptr)
      throw Logic_exception("eek: %p", buffer);
    
    assert(wr);
    work_key = wr->work_key();
    auto response_vector = wr->response();
    assert(response_vector);
    response_len = response_vector->Length();
    response = std::malloc(response_len);
    status = wr->status();
    memcpy(response, response_vector->Data(), response_len);
  }

  free_buffer(buffer);
  return true;
}

status_t ADO_protocol_builder::recv_bootstrap_response()
{

  Buffer_header * buffer;

  poll_recv(buffer);

  /* TODO: check bootstrap response */
  free_buffer(buffer);
  return S_OK;
}

void ADO_protocol_builder::send_table_op_create(const uint64_t work_request_id,
                                                const std::string& keystr,
                                                const size_t value_len)
{
  Flatbuffer_builder fbb(this);

  auto key = fbb.CreateString(keystr);
  auto tor = CreateTableOpRequest(fbb, work_request_id, TableOpType_Create, value_len, 0, 0, key);
  auto msg = CreateMessage(fbb, Element_TableOpRequest, tor.Union());
  fbb.Finish(msg);

  send_callback(fbb.release());
}

void ADO_protocol_builder::send_table_op_open(const uint64_t work_request_id,
                                              const std::string& keystr)
{
  Flatbuffer_builder fbb(this);

  auto key = fbb.CreateString(keystr);
  auto tor = CreateTableOpRequest(fbb, work_request_id, TableOpType_Open, 0, 0, 0, key);
  auto msg = CreateMessage(fbb, Element_TableOpRequest, tor.Union());
  fbb.Finish(msg);

  send_callback(fbb.release());
}

void ADO_protocol_builder::send_table_op_resize(const uint64_t work_request_id,
                                                const std::string& keystr,
                                                const size_t new_value_len)
{
  Flatbuffer_builder fbb(this);

  auto key = fbb.CreateString(keystr);
  auto tor = CreateTableOpRequest(fbb, work_request_id, TableOpType_ValueResize, new_value_len, 0, 0, key);
  auto msg = CreateMessage(fbb, Element_TableOpRequest, tor.Union());
  fbb.Finish(msg);

  send_callback(fbb.release());
}

void ADO_protocol_builder::send_table_op_erase(const uint64_t work_request_id,
                                               const std::string& keystr)
{
  Flatbuffer_builder fbb(this);

  auto key = fbb.CreateString(keystr);
  auto tor = CreateTableOpRequest(fbb, work_request_id, TableOpType_Erase, 0, 0, 0, key);
  auto msg = CreateMessage(fbb, Element_TableOpRequest, tor.Union());
  fbb.Finish(msg);

  send_callback(fbb.release());
}

// table TableOpRequest
// {
//   op        : TableOpType;
//   key       : string;
//   value_len : uint64;
//   work_key  : uint64;
// }

// table TableOpMemory
// {
//   work_key      : uint64;
//   addr_or_align : uint64;
//   len           : uint64;
// }


void ADO_protocol_builder::send_table_op_allocate_pool_memory(const uint64_t work_request_id,
                                                              const size_t size,
                                                              const size_t alignment)
{
  Flatbuffer_builder fbb(this);
  
  auto tom = CreateTableOpRequest(fbb, work_request_id, TableOpType_AllocatePoolMemory, size, 0, alignment);
  auto msg = CreateMessage(fbb, Element_TableOpRequest, tom.Union());
  fbb.Finish(msg);

  send_callback(fbb.release());
}

void ADO_protocol_builder::send_table_op_free_pool_memory(const uint64_t work_request_id,
                                                          const void * ptr,
                                                          const size_t size)
{
  Flatbuffer_builder fbb(this);
  
  auto tom = CreateTableOpRequest(fbb, work_request_id, TableOpType_FreePoolMemory, size,
                                  reinterpret_cast<uint64_t>(ptr));
  auto msg = CreateMessage(fbb, Element_TableOpRequest, tom.Union());
  fbb.Finish(msg);

  send_callback(fbb.release());
}




void ADO_protocol_builder::send_table_op_response(const status_t status,
                                                  const void * value_addr,
                                                  size_t value_len)
{
  Flatbuffer_builder fbb(this);

  //  auto topr = CreateTableOpResponse(fbb, status, reinterpret_cast<uint64_t>(value_addr), value_len);

  TableOpResponseBuilder builder_(fbb);

  builder_.add_status(status);

  if(value_addr)
    builder_.add_value_addr(reinterpret_cast<uint64_t>(value_addr));

  if(value_len > 0)
    builder_.add_value_len(value_len);

  auto topr = builder_.Finish();
  auto msg = CreateMessage(fbb, Element_TableOpResponse, topr.Union());
  fbb.Finish(msg);

  send_callback(fbb.release());
}

void ADO_protocol_builder::recv_table_op_response(status_t& status,
                                                  void *& value_addr,
                                                  size_t * p_out_value_len)
{
  Buffer_header * buffer;
  poll_recv_callback(buffer);

  auto protocol_start = buffer->to_message();
  auto msg = GetMessage(protocol_start);
  auto topr = msg->element_as_TableOpResponse();
  status = topr->status();

  if(p_out_value_len)
    *p_out_value_len = topr->value_len();

  value_addr = reinterpret_cast<void*>(topr->value_addr());

  free_buffer(buffer);
}

bool ADO_protocol_builder::recv_table_op_request(uint64_t& work_request_id,
                                                 int& op,
                                                 std::string& key,
                                                 size_t& value_len,
                                                 size_t& value_alignment,
                                                 void* & addr)
{
  Buffer_header * buffer;
  status_t s = recv_callback(buffer);

  if(s == E_EMPTY) {
    return false;
  }
  else if(s == S_OK) {
    auto protocol_start = buffer->to_message();
    auto msg = GetMessage(protocol_start);    
    auto topr = msg->element_as_TableOpRequest();

    if(topr) {
      op = topr->op();
      work_request_id = topr->work_key();
      if(topr->key())
        key = topr->key()->c_str();
      value_len = topr->value_len();
      value_alignment = topr->align();
      addr = reinterpret_cast<void*>(topr->addr());
    }
    else throw Logic_exception("recv_table_op_request: bad protocol");
  }
  else assert(0);

  free_buffer(buffer);
  return true;
}





