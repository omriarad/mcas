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
#ifndef __MCAS_CLIENT_HANDLER_H__
#define __MCAS_CLIENT_HANDLER_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Weffc++"

#include "fabric_transport.h"
#include "mcas_client_config.h"
#include "protocol.h"
#include "protocol_ostream.h"

#include <api/fabric_itf.h>
#include <api/mcas_itf.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/byte_buffer.h>
#include <common/perf/tm.h>
#include <common/string_view.h>
#include <gsl/pointers>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <gnutls/gnutls.h>
#include <gnutls/x509.h>
#include <gnutls/crypto.h>

#include <boost/numeric/conversion/cast.hpp>
#include <map>
#include <set>
#include <tuple>

/* Enable this to introduce locks to prevent re-entry by multiple
   threads.  The client is not re-entrant because of the state machine
   and multi-packet operations
*/
#define THREAD_SAFE_CLIENT

#define CAFILE "/etc/ssl/certs/ca-bundle.trust.crt"

#ifdef THREAD_SAFE_CLIENT
#define API_LOCK() std::lock_guard<std::mutex> g(_api_lock);
#else
#define API_LOCK()
#endif

namespace mcas
{

namespace client
{

struct TLS_transport
{
  static unsigned debug_level();
  static ssize_t gnutls_pull_func(gnutls_transport_ptr_t, void*, size_t);
  static ssize_t gnutls_vec_push_func(gnutls_transport_ptr_t, const giovec_t * , int );
  static int gnutls_pull_timeout_func(gnutls_transport_ptr_t, unsigned int);
};



struct async_buffer_set_t;
struct iob_free;

static const char *env_scbe = ::getenv("SHORT_CIRCUIT_BACKEND");

/* Adaptor point for other transports */
using Connection_base = mcas::client::Fabric_transport;

/**
 * Client side connection handler
 *
 */
class Connection_handler : public Connection_base {
  using iob_ptr        = std::unique_ptr<client::Fabric_transport::buffer_t, iob_free>;
  using locate_element = protocol::Message_IO_response::locate_element;

public:
  friend struct TLS_transport;

  using memory_region_t = typename Transport::memory_region_t;
  template <typename T>
    using basic_string_view = common::basic_string_view<T>;
  using byte = common::byte;

  /**
   * Constructor
   *
   * @param debug_level
   * @param connection
   * @param patience Time to wait (in seconds) for single fabric post to complete
   * @param other Additional configuration string
   *
   * @return
   */
  Connection_handler(const unsigned debug_level,
                     Connection_base::Transport *connection,
                     Connection_base::buffer_manager &bm,
                     const unsigned patience,
                     common::string_view other);

  ~Connection_handler();

private:
  enum State {
              INITIALIZE,
              HANDSHAKE_SEND,
              HANDSHAKE_GET_RESPONSE,
              SHUTDOWN,
              STOPPED,
              READY,
  };

  State _state = State::INITIALIZE;

  template <typename MT>
    status_t invoke_ado_common(
      const iob_ptr & iobs
      , const MT *msg
      , std::vector<component::IMCAS::ADO_response>& out_response
      , unsigned int flags
    );

public:
  using pool_t = uint64_t;

  void bootstrap()
  {
    set_state(INITIALIZE);
    while (tick() > 0)
      ;
  }

  void shutdown()
  {
    set_state(SHUTDOWN);
    while (tick() > 0) sleep(1);
  }

  pool_t open_pool(const std::string name,
                   const unsigned int flags,
                   const addr_t base);

  pool_t create_pool(const std::string  name,
                     const size_t       size,
                     const unsigned int flags,
                     const uint64_t     expected_obj_count,
                     const addr_t       base);

  status_t close_pool(const pool_t pool);

  status_t delete_pool(const std::string &name);

  status_t delete_pool(const pool_t pool);

  status_t configure_pool(const component::IKVStore::pool_t pool, const std::string &json);

  status_t put(const pool_t       pool,
               const std::string  key,
               const void *       value,
               const size_t       value_len,
               component::IMCAS::flags_t flags);

  status_t put(const pool_t       pool,
               const void *       key,
               const size_t       key_len,
               const void *       value,
               const size_t       value_len,
               component::IMCAS::flags_t flags);

  status_t put_direct(pool_t                               pool,
                      const void *                         key,
                      size_t                               key_len,
                      gsl::span<const common::const_byte_span> values,
                      component::Registrar_memory_direct * rmd,
                      gsl::span<const component::IMCAS::memory_handle_t> handles,
                      component::IMCAS::flags_t            flags);

  status_t async_put(const pool_t                      pool,
                     const void *                      key,
                     const size_t                      key_len,
                     const void *                      value,
                     size_t                            value_len,
                     component::IMCAS::async_handle_t &out_handle,
                     component::IMCAS::flags_t         flags);

  status_t async_put_direct(const component::IMCAS::pool_t       pool,
                            const void *                         key,
                            size_t                               key_len,
                            gsl::span<const common::const_byte_span> values,
                            component::IMCAS::async_handle_t &   out_handle,
                            component::Registrar_memory_direct * rmd,
                            gsl::span<const component::IKVStore::memory_handle_t> handles,
                            component::IMCAS::flags_t            flags);

  status_t async_get_direct(TM_FORMAL const component::IMCAS::pool_t       pool,
                            const void *                         key,
                            size_t                               key_len,
                            void *                               value,
                            size_t &                             value_len,
                            component::IMCAS::async_handle_t &   out_handle,
                            component::Registrar_memory_direct * rmd,
                            component::IKVStore::memory_handle_t handle,
                            component::IMCAS::flags_t            flags);

  status_t check_async_completion(component::IMCAS::async_handle_t &handle);

  status_t get(const pool_t pool, const std::string &key, std::string &value);

  status_t get(const pool_t pool, const std::string &key, void *&value, size_t &value_len);

  status_t get_direct(const pool_t                         pool,
                      const void *                         key,
                      size_t                               key_len,
                      void *                               value,
                      size_t &                             out_value_len,
                      component::Registrar_memory_direct * rmd,
                      component::IKVStore::memory_handle_t handle = component::IKVStore::HANDLE_NONE);

  status_t get_direct_offset(pool_t                              pool,
                             std::size_t                         offset,
                             std::size_t &                       length,
                             void *                              buffer,
                             component::Registrar_memory_direct *rmd,
                             component::IMCAS::memory_handle_t   handle);

  status_t async_get_direct_offset(pool_t                              pool,
                                   std::size_t                         offset,
                                   std::size_t &                       length,
                                   void *                              buffer,
                                   component::IMCAS::async_handle_t &  out_handle,
                                   component::Registrar_memory_direct *rmd,
                                   component::IMCAS::memory_handle_t   handle);

  status_t put_direct_offset(pool_t                              pool,
                             std::size_t                         offset,
                             std::size_t &                       length,
                             const void *                        buffer,
                             component::Registrar_memory_direct *rmd,
                             component::IMCAS::memory_handle_t   handle);

  status_t async_put_direct_offset(pool_t                              pool,
                                   std::size_t                         offset,
                                   std::size_t &                       length,
                                   const void *                        buffer,
                                   component::IMCAS::async_handle_t &  out_handle,
                                   component::Registrar_memory_direct *rmd,
                                   component::IMCAS::memory_handle_t   handle);

  status_t erase(const pool_t pool, const std::string &key);

  status_t async_erase(const component::IMCAS::pool_t    pool,
                       const std::string &               key,
                       component::IMCAS::async_handle_t &out_handle);

  uint64_t key_hash(const void *key, const size_t key_len);

  uint64_t auth_id() const
  {
    /* temporary */
    auto     env = getenv("MCAS_AUTH_ID");
    uint64_t auth_id;
    if (env) {
      auto id = std::strtoll(env, nullptr, 10);
      auth_id = boost::numeric_cast<uint64_t>(id);
    }
    else {
      auth_id = boost::numeric_cast<uint64_t>(getuid());
    }

    return auth_id;
  }

  size_t count(const pool_t pool);

  status_t get_attribute(const component::IKVStore::pool_t    pool,
                         const component::IKVStore::Attribute attr,
                         std::vector<uint64_t> &              out_attr,
                         const std::string *                  key);

  status_t get_statistics(component::IMCAS::Shard_stats &out_stats);

  status_t find(const component::IKVStore::pool_t pool,
                const std::string &               key_expression,
                const offset_t                    offset,
                offset_t &                        out_matched_offset,
                std::string &                     out_matched_key);

  status_t invoke_ado(const component::IMCAS::pool_t               pool,
                      basic_string_view<byte>                      key,
                      basic_string_view<byte>                      request,
                      const unsigned int                           flags,
                      std::vector<component::IMCAS::ADO_response> &out_response,
                      const size_t                                 value_size);

  status_t invoke_ado_async(const component::IMCAS::pool_t               pool,
                            basic_string_view<byte>                      key,
                            basic_string_view<byte>                      request,
                            const component::IMCAS::ado_flags_t          flags,
                            std::vector<component::IMCAS::ADO_response> &out_response,
                            component::IMCAS::async_handle_t &           out_async_handle,
                            const size_t                                 value_size);

  status_t invoke_put_ado(const component::IMCAS::pool_t               pool,
                          basic_string_view<byte>                      key,
                          basic_string_view<byte>                      request,
                          basic_string_view<byte>                      value,
                          size_t                                       root_len,
                          const unsigned int                           flags,
                          std::vector<component::IMCAS::ADO_response> &out_response);

  status_t invoke_put_ado_async(const component::IMCAS::pool_t                  pool,
                                const basic_string_view<byte>                   key,
                                const basic_string_view<byte>                   request,
                                const basic_string_view<byte>                   value,
                                const size_t                                    root_len,
                                const component::IMCAS::ado_flags_t             flags,
                                std::vector<component::IMCAS::ADO_response>&    out_response,
                                component::IMCAS::async_handle_t&               out_async_handle);

  bool check_message_size(size_t size) const { return size > _max_message_size; }

  status_t receive_and_process_ado_response(
    const iob_ptr & iobr
    , std::vector<component::IMCAS::ADO_response> & out_response
  );

private:
  /**
   * FSM tick call
   *
   */
  int tick();

  /**
   * FSM state change
   *
   * @param s State to change to
   */
  inline void set_state(State s)
  {
    if (2 < debug_level()) {
      static const std::map<State, const char *> m{
                                                   {INITIALIZE, "INITIALIZE"},
                                                   {HANDSHAKE_SEND, "HANDSHAKE_SEND"},
                                                   {HANDSHAKE_GET_RESPONSE, "HANDSHAKE_GET_RESPONSE"},
                                                   {SHUTDOWN, "SHUTDOWN"},
                                                   {STOPPED, "STOPPED"},
                                                   {READY, "READY"},
      };
      PLOG("Client state %s -> %s", m.find(_state)->second, m.find(s)->second);
    }
    _state = s;
  } /* we could add transition checking later */

  void start_tls();

  template <typename MT>
  void msg_send_log(const MT *m, const void *context, const char *desc) { msg_send_log(1, m, context, desc); }

  template <typename MT>
  void msg_send_log(const unsigned level, const MT *msg, const void *context, const char *desc)
  {
    msg_log(level, msg, context, desc, "SEND");
  }

  template <typename MT>
  void msg_log(const unsigned level_, const MT *msg_, const void *context_, const char *desc_, const char *direction_)
  {
    if ( level_ < debug_level() )
      {
        std::ostringstream m;
        m << *msg_;
        PLOG("%s (%p) (%s) %s", direction_, context_, desc_, m.str().c_str());
      }
  }

  template <typename MT>
  void msg_recv_log(const MT *m, const void *context, const char *desc) { msg_recv_log(1, m, context, desc); }

  template <typename MT>
  void msg_recv_log(const unsigned level, const MT *msg, const void *context, const char *desc)
  {
    msg_log(level, msg, context, desc, "RECV");
  }

public:

  template <typename MT>
  gsl::not_null<const MT *>msg_recv(const buffer_t *iob, const char *desc)
  {
    /*
     * First, cast the response buffer to Message (checking version).
     * Second, cast the Message to a specific message type (checking message type).
     */
    const auto *const msg = mcas::protocol::message_cast(iob->base());
    const auto *const response_msg = msg->ptr_cast<MT>();
    msg_recv_log(response_msg, iob, desc);
    return response_msg;
  }

  template <typename MT>
  void sync_inject_send(buffer_t *iob, const MT *msg, std::size_t size, const char *desc)
  {
    msg_send_log(msg, iob, desc);
    Connection_base::sync_inject_send(iob, size);
  }

  template <typename MT>
  void sync_inject_send(buffer_t *iob, const MT *msg, const char *desc)
  {
    sync_inject_send(iob, msg, msg->msg_len(), desc);
  }

  template <typename MT>
  void sync_send(buffer_t *iob, const MT *msg, const char *desc)
  {
    msg_send_log(msg, iob, desc);
    Connection_base::sync_send(iob);
  }

  void sync_inject_send(buffer_t *iob, std::size_t size)
  {
    Connection_base::sync_inject_send(iob, size);
  }


private:
  /* unused */
  template <typename MT>
  void post_send(buffer_t *iob, const MT *msg, const char *desc)
  {
    msg_send_log(2, msg, iob, desc);
    Connection_base::post_send(iob);
  }

  template <typename MT>
  void post_send(const ::iovec *first, const ::iovec *last, void **descriptors, void *context, const MT *msg, const char *desc)
  {
    msg_send_log(msg, context, desc);
    Connection_base::post_send(first, last, descriptors, context);
  }

  /**
   * Put used when the value exceeds the size of the basic
   * IO buffer (e.g., 2MB).  This version of put will perform
   * a two-stage exchange, advance notice and then value
   *
   * @param pool Pool identifier
   * @param key Key
   * @param key_len Key length
   * @param value_len Value length
   * @param flags ignored?
   *
   * @return
   */
  std::tuple<uint64_t, uint64_t> put_locate(const pool_t   pool,
                                            const void *   key,
                                            const size_t   key_len,
                                            const size_t   value_len,
                                            const unsigned flags);

  component::IMCAS::async_handle_t put_locate_async(TM_FORMAL pool_t                    pool,
                                                    const void *                        key,
                                                    size_t                              key_len,
                                                    gsl::span<const common::const_byte_span> values,
                                                    component::Registrar_memory_direct *rmd,
                                                    gsl::span<const component::IKVStore::memory_handle_t> mem_handles_,
                                                    unsigned                            flags);

  std::tuple<uint64_t, uint64_t, std::size_t> get_locate(const pool_t   pool,
                                                         const void *   key,
                                                         const size_t   key_len,
                                                         const unsigned flags);

  component::IMCAS::async_handle_t get_locate_async(TM_FORMAL pool_t                     pool,
                                                    const void *                        key,
                                                    size_t                              key_len,
                                                    void *                              value,
                                                    size_t &                            value_len,
                                                    component::Registrar_memory_direct *rmd,
                                                    void *                              desc,
                                                    unsigned                            flags);
public:
  iob_ptr make_iob_ptr(buffer_t::completion_t);
  iob_ptr make_iob_ptr_recv();
  iob_ptr make_iob_ptr_send();
  iob_ptr make_iob_ptr_write();
  iob_ptr make_iob_ptr_read();

private:
  static void send_complete(void *, buffer_t *iob);
  static void recv_complete(void *, buffer_t *iob);
  static void write_complete(void *, buffer_t *iob);
  static void read_complete(void *, buffer_t *iob);

  std::tuple<uint64_t, std::vector<locate_element>> locate(pool_t pool, std::size_t offset, std::size_t size);

  component::IMCAS::async_handle_t get_direct_offset_async(pool_t                              pool,
                                                           std::size_t                         offset,
                                                           void *                              buffer,
                                                           std::size_t &                       length,
                                                           component::Registrar_memory_direct *rmd,
                                                           void *                              desc);

  component::IMCAS::async_handle_t put_direct_offset_async(pool_t                              pool,
                                                           std::size_t                         offset,
                                                           const void *                        buffer,
                                                           std::size_t &                       length,
                                                           component::Registrar_memory_direct *rmd,
                                                           void *                              desc);

private:
#ifdef THREAD_SAFE_CLIENT
  std::mutex _api_lock;
#endif

  bool     _exit;
  bool     _force_direct = false;
  uint64_t _request_id;

public: /* for async "move_along" processing */
  uint64_t request_id() { return ++_request_id; }

private:
  size_t _max_message_size;
  size_t _max_inject_size;

  struct options_s {
    bool short_circuit_backend;
    unsigned tls   : 1;
    unsigned hmac : 1;

    options_s()
      : short_circuit_backend(env_scbe && env_scbe[0] == '1'), tls(0), hmac(0)
    {}
  };

  options_s _options;

  gnutls_certificate_credentials_t _xcred;
  gnutls_priority_t                _priority;
  gnutls_session_t                 _session;
  common::Byte_buffer              _tls_buffer;

};

}  // namespace client
}  // namespace mcas

#pragma GCC diagnostic pop
#endif
