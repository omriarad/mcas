/*
   Copyright [2017-2020] [IBM Corporation]
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
#include "mcas_client.h"

#include <api/fabric_itf.h>
#include <common/json.h>
#include <city.h>

#include "connection.h"
#include "protocol.h"

#include <regex>
#include <vector>

using namespace component;

namespace mcas
{
namespace global
{
unsigned debug_level = 0;
}
}  // namespace mcas

MCAS_client::MCAS_client(const unsigned                      debug_level,
                         const common::string_view           src_device,
                         const common::string_view           src_addr,
                         const common::string_view           provider,
                         const common::string_view           dest_addr,
                         std::uint16_t                       port,
                         const unsigned                      patience_,
                         const common::string_view           other_)
: common::log_source(debug_level),
  _factory(load_factory()),
  _fabric(make_fabric_sip(*_factory, src_addr, src_device, provider)),
  _ep(_fabric->make_endpoint(common::json::serializer<common::json::dummy_writer>::object{}.str(), dest_addr, port)),
  _bm(debug_level, _ep.get()),
  _transport(_ep->make_open_client()),
  _connection(std::make_unique<mcas::client::Connection_handler>(debug_level, _transport.get(), _bm, patience_, other_)),
  _open_connection(*_connection)
{
  CPLOG(3, "Extra config: %s", other_.data());
}

Open_connection::Open_connection(mcas::client::Connection_handler &_connection)
    : _open_cnxn((_connection.bootstrap(), &_connection))
{
}

Open_connection::~Open_connection()
{
  if (_open_cnxn) {
    _open_cnxn->shutdown();
  }
}

auto MCAS_client::load_factory() -> IFabric_factory *
{
  IBase *comp = load_component("libcomponent-fabric.so", net_fabric_factory);

  if (!comp)
    throw General_exception("Fabric component not found");

  auto factory = static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid()));
  assert(factory);
  return factory;
}

/* make_fabric: source/IP/provider form */
auto MCAS_client::make_fabric_sip(component::IFabric_factory &        factory_,
                              const common::string_view src_addr_,
                              const common::string_view domain_name_,
                              const common::string_view fabric_prov_name_) -> IFabric *
{
  namespace c_json = common::json;
  using json = c_json::serializer<c_json::dummy_writer>;
  auto fabric_spec =
    json::object(
      json::member("ep_attr", json::object(json::member("type", "FI_EP_MSG")))
    );

    if ( fabric_prov_name_.data() )
    {
      fabric_spec
        .append(
          json::member("fabric_attr", json::object(json::member("prov_name", fabric_prov_name_)))
        )
        ;
    }

    if ( src_addr_.data() )
    {
      fabric_spec
        .append(
          json::member("addr_format", "FI_ADDR_STR")
        )
        .append(
          json::member("src_addr", "fi_sockaddr_in://" + std::string(src_addr_) + ":0")
        )
        ;
    }

    if ( domain_name_.data() )
    {
      fabric_spec
        .append(
          json::member(
            "domain_attr"
            , json::object(
                json::member("name", domain_name_)
                , json::member("threading", "FI_THREAD_SAFE")
            )
          )
        )
        ;
    }

  return factory_.make_fabric(fabric_spec.str());
}

/* make_fabric: address/prover/device form */
auto MCAS_client::make_fabric_apd(component::IFabric_factory &factory_,
                              const common::string_view  // ip_addr
                              ,
                              const common::string_view provider,
                              const common::string_view device) -> IFabric *
{
  namespace c_json = common::json;
  using json = c_json::serializer<c_json::dummy_writer>;
  auto mr_mode =
    json::array(
      provider == "sockets"
      ? json::array(
        "FI_MR_VIRT_ADDR"
        , "FI_MR_ALLOCATED"
        , "FI_MR_PROV_KEY"
      )
      : json::array(
          "FI_MR_LOCAL"
        , "FI_MR_VIRT_ADDR"
        , "FI_MR_ALLOCATED"
        , "FI_MR_PROV_KEY"
      )
    )
    ;

  auto fabric_spec =
    json::object(
      json::member("fabric_attr", json::object(json::member("prov_name", provider)))
      , json::member(
          "domain_attr"
          , json::object(
              json::member("mr_mode", std::move(mr_mode))
              , json::member("name", device)
            )
        )
      , json::member("ep_attr", json::object(json::member("type", "FI_EP_MSG")))
    );
  return factory_.make_fabric(fabric_spec.str());
}

int MCAS_client::thread_safety() const { return IKVStore::THREAD_MODEL_SINGLE_PER_POOL; }

int MCAS_client::get_capability(Capability cap) const
{
  switch (cap) {
    case Capability::POOL_DELETE_CHECK:
      return 1;
    case Capability::POOL_THREAD_SAFE:
      return 1;
    case Capability::RWLOCK_PER_POOL:
      return 1;
    default:
      return -1;
  }
}

IKVStore::pool_t MCAS_client::create_pool(const std::string &  name,
                                          const size_t         size,
                                          const uint32_t       flags,
                                          const uint64_t       expected_obj_count,
                                          const IKVStore::Addr base)
{
  return _connection->create_pool(name, size, flags, expected_obj_count, base.addr);
}

IKVStore::pool_t MCAS_client::open_pool(const std::string &  name,
                                        const uint32_t       flags,
                                        const IKVStore::Addr base)
{
  return _connection->open_pool(name, flags, base.addr);
}

status_t MCAS_client::close_pool(const IKVStore::pool_t pool)
{
  if (!pool) return E_INVAL;
  return _connection->close_pool(pool);
}

status_t MCAS_client::delete_pool(const std::string &name)
{
  return _connection->delete_pool(name);
}

status_t MCAS_client::delete_pool(IKVStore::pool_t pool)
{
  return _connection->delete_pool(pool);
}

status_t MCAS_client::configure_pool(const IKVStore::pool_t pool, const std::string &json)
{
  return _connection->configure_pool(pool, json);
}

status_t MCAS_client::put(const IKVStore::pool_t pool,
                          const std::string &    key,
                          const void *           value,
                          const size_t           value_len,
                          uint32_t               flags)
{
  assert(flags <= IMCAS::FLAGS_MAX_VALUE);
  return _connection->put(pool, key, value, value_len, flags);
}

status_t MCAS_client::put_direct(const pool_t           pool,
                                 const std::string &    key,
                                 gsl::span<const common::const_byte_span> values,
                                 gsl::span<const IMCAS::memory_handle_t> handles,
                                 const flags_t          flags)
{
  return _connection->put_direct(pool, key.data(), key.size(), values, registrar(), handles, flags);
}

status_t MCAS_client::async_put(IKVStore::pool_t   pool,
                                const std::string &key,
                                const void *       value,
                                size_t             value_len,
                                async_handle_t &   out_handle,
                                const flags_t      flags)
{
  return _connection->async_put(pool, key.data(), key.size(), value, value_len, out_handle, flags);
}

status_t MCAS_client::async_put_direct(const IKVStore::pool_t          pool,
                                       const std::string &             key,
                                       gsl::span<const common::const_byte_span> values,
                                       async_handle_t &                out_handle,
                                       gsl::span<const IMCAS::memory_handle_t> handles,
                                       const flags_t                   flags)
{
  return _connection->async_put_direct(pool, key.data(), key.size(), values, out_handle, registrar(), handles, flags);
}

status_t MCAS_client::async_get_direct(IKVStore::pool_t          pool,
                                       const std::string &       key,
                                       void *                    value,
                                       size_t &                  value_len,
                                       async_handle_t &          out_handle,
                                       IKVStore::memory_handle_t handle)
{
  TM_ROOT();
  return _connection->async_get_direct(TM_REF pool, key.data(), key.size(), value, value_len, out_handle, registrar(), handle, 0);
}

status_t MCAS_client::check_async_completion(async_handle_t &handle)
{
  TM_ROOT();
  return _connection->check_async_completion(handle);
}

status_t MCAS_client::get(const IKVStore::pool_t pool,
                          const std::string &    key,
                          void *&                out_value, /* release with free() */
                          size_t &               out_value_len)
{
  return _connection->get(pool, key, out_value, out_value_len);
}

status_t MCAS_client::get_direct(const pool_t           pool,
                                 const std::string &    key,
                                 void *                 out_value,
                                 size_t &               out_value_len,
                                 IMCAS::memory_handle_t handle)
{
  return _connection->get_direct(pool, key.data(), key.size(), out_value, out_value_len, registrar(), handle);
}

status_t MCAS_client::get_direct_offset(const IMCAS::pool_t          pool,
                                        const offset_t               offset,
                                        size_t &                     length,
                                        void *const                  out_buffer,
                                        const IMCAS::memory_handle_t handle)
{
  return _connection->get_direct_offset(pool, offset, length, out_buffer, registrar(), handle);
}

status_t MCAS_client::async_get_direct_offset(const IMCAS::pool_t          pool,
                                              const offset_t               offset,
                                              size_t &                     length,
                                              void *const                  out_buffer,
                                              async_handle_t &             out_handle,
                                              const IMCAS::memory_handle_t handle)
{
  return _connection->async_get_direct_offset(pool, offset, length, out_buffer, out_handle, registrar(), handle);
}

status_t MCAS_client::put_direct_offset(const IMCAS::pool_t          pool,
                                        const offset_t               offset,
                                        size_t &                     length,
                                        const void *const            out_buffer,
                                        const IMCAS::memory_handle_t handle)
{
  return _connection->put_direct_offset(pool, offset, length, out_buffer, registrar(), handle);
}

status_t MCAS_client::async_put_direct_offset(const IMCAS::pool_t          pool,
                                              const offset_t               offset,
                                              size_t &                     length,
                                              const void *const            out_buffer,
                                              async_handle_t &             out_handle,
                                              const IMCAS::memory_handle_t handle)
{
  return _connection->async_put_direct_offset(pool, offset, length, out_buffer, out_handle, registrar(), handle);
}

component::IKVStore::memory_handle_t MCAS_client::register_direct_memory(common::const_byte_span mem)
{
  /* register_direct_memory is exposed at the interface. A failure of madvise
   * might well be expected if the parameters passed in at the interface are
   * incompatible with madvise, e.g. vaddr is not page-aligned. That might be
   * fixed by using On Demand Paging (ODP), which should remove the need for
   * madvise.
   */
  if (madvise(const_cast<void *>(::base(mem)), ::size(mem), MADV_DONTFORK) != 0) {
    if (debug_level() > 2)
      PWRN("MCAS_client::%s: madvise MADV_DONTFORK failed (%p %lu) %s", __func__, ::base(mem), ::size(mem), strerror(errno));
  }

  return _connection->register_direct_memory(mem);
}

status_t MCAS_client::unregister_direct_memory(IKVStore::memory_handle_t handle)
{
  return _connection->unregister_direct_memory(handle);
}

status_t MCAS_client::erase(const IKVStore::pool_t pool, const std::string &key)
{
  return _connection->erase(pool, key);
}

status_t MCAS_client::async_erase(const IMCAS::pool_t pool, const std::string &key, async_handle_t &out_handle)
{
  return _connection->async_erase(pool, key, out_handle);
}

size_t MCAS_client::count(const IKVStore::pool_t pool) { return _connection->count(pool); }

status_t MCAS_client::get_attribute(const IKVStore::pool_t    pool,
                                    const IKVStore::Attribute attr,
                                    std::vector<uint64_t> &   out_attr,
                                    const std::string *       key)
{
  return _connection->get_attribute(pool, attr, out_attr, key);
}

status_t MCAS_client::get_statistics(Shard_stats &out_stats) { return _connection->get_statistics(out_stats); }

status_t MCAS_client::free_memory(void *p)
{
  ::free(p);
  return S_OK;
}

void MCAS_client::debug(const IKVStore::pool_t  // pool
                        ,
                        unsigned  // cmd
                        ,
                        uint64_t  // arg
)
{
}

status_t MCAS_client::find(const IKVStore::pool_t pool,
                           const std::string &    key_expression,
                           const offset_t         offset,
                           offset_t &             out_matched_offset,
                           std::string &          out_matched_key)
{
  return _connection->find(pool, key_expression, offset, out_matched_offset, out_matched_key);
}

status_t MCAS_client::invoke_ado(const IKVStore::pool_t            pool,
                                 basic_string_view<byte>           key,
                                 basic_string_view<byte>           request,
                                 const uint32_t                    flags,
                                 std::vector<IMCAS::ADO_response> &out_response,
                                 const size_t                      value_size)
{
  return _connection->invoke_ado(pool, key, request, flags, out_response, value_size);
}

status_t MCAS_client::async_invoke_ado(const IMCAS::pool_t        pool,
                                       basic_string_view<byte>    key,
                                       basic_string_view<byte>    request,
                                       const ado_flags_t          flags,
                                       std::vector<ADO_response> &out_response,
                                       async_handle_t &           out_async_handle,
                                       const size_t               value_size)
{
  return _connection->invoke_ado_async(pool, key, request, flags, out_response, out_async_handle,
                                       value_size);
}

status_t MCAS_client::invoke_put_ado(const IKVStore::pool_t            pool,
                                     basic_string_view<byte>           key,
                                     basic_string_view<byte>           request,
                                     basic_string_view<byte>           value,
                                     size_t                            root_len,
                                     ado_flags_t                       flags,
                                     std::vector<IMCAS::ADO_response> &out_response)
{
  return _connection->invoke_put_ado(pool, key, request, value, root_len, flags, out_response);
}

status_t MCAS_client::async_invoke_put_ado(const IMCAS::pool_t           pool,
                                           const basic_string_view<byte> key,
                                           const basic_string_view<byte> request,
                                           const basic_string_view<byte> value,
                                           const size_t                  root_len,
                                           const ado_flags_t             flags,
                                           std::vector<ADO_response>&    out_response,
                                           async_handle_t&               out_async_handle)
{
  return _connection->invoke_put_ado_async(pool, key, request, value, root_len, flags, out_response, out_async_handle);
}


/**
 * Factory entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t component_id)
{
  if (component_id == MCAS_client_factory::component_id()) {
    auto fact = new MCAS_client_factory();
    fact->add_ref();
    return static_cast<void *>(fact);
  }
  else {
    PWRN("%s: request for bad factory type", __func__);
    return NULL;
  }
}
