#ifndef __REST_ENDPOINT__
#define __REST_ENDPOINT__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <pistache/endpoint.h>
#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/description.h>
#include <pistache/serializer/rapidjson.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <rapidjson/document.h>
#include <common/utils.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <set>
#include <city.h>

#include "rest_server_config.h"

using namespace Pistache;

namespace Generic
{
void handle_ready(const Rest::Request&, Http::ResponseWriter response);
}

class Backend
{
public:
  Backend(const std::string& path,
          const unsigned debug_level = 0,
          const std::string& store = "hstore")
  {
    using namespace component;

    std::string store_lib = "libcomponent-" + store + ".so";
    IBase * comp;


    if(store == "hstore-cc" || store == "hstore") { /* HSTORE-CC */
      comp = load_component(store_lib.c_str(), component::hstore_factory);
      assert(comp);
      
      auto fact = static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid()));
      
      std::stringstream ss;
      ss << "[{\"path\":\"" << path << "\",\"addr\":" << _load_addr << "}]";
      PLOG("dax config: %s", ss.str().c_str());

      _itf = make_itf_ref(fact->create(debug_level,
                                       {
                                        {+component::IKVStore_factory::k_debug, std::to_string(debug_level)},
                                        {+component::IKVStore_factory::k_dax_config, ss.str()}
                                       }));
      fact->release_ref();
    }
    else if(store == "mapstore") { /* MAPSTORE */
      comp = load_component(store_lib.c_str(), component::mapstore_factory);
      assert(comp);
      
      auto fact = static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid()));

      _itf = make_itf_ref(fact->create(debug_level,
                                       {
                                        {+component::IKVStore_factory::k_debug, std::to_string(debug_level)},
                                       }));
      fact->release_ref();
    }
    else throw API_exception("unabled store type");
  }

  inline void close_pool(const component::IKVStore::pool_t pool_handle) {
    _itf->close_pool(pool_handle);
  }

  auto kvstore() {
    return _itf.get();
  }
  
private:
  static constexpr uint64_t               _load_addr = 0x8800000000;

protected:
  component::Itf_ref<component::IKVStore> _itf;
};

using session_id_t = std::string;
using client_id_t = std::string;

class Pool_manager : private Backend
{
public:
  
  struct Session {
    component::IKVStore::pool_t pool_handle;
  };

  Pool_manager(const std::string& pmem_path, const unsigned debug_level) : Backend(pmem_path, debug_level) {
  }

  status_t create_or_open_pool(const std::string& client_id, const std::string& pool_name, const size_t size_mb, session_id_t& session_id) {
    auto pool = _itf->open_pool(pool_name);
    if(pool != component::IKVStore::POOL_ERROR) {
      PLOG("endpoint: opened existing pool");
    }
    else {
      PLOG("endpoint: creating new pool");
      pool = _itf->create_pool(pool_name, MiB(size_mb));
      if(pool == component::IKVStore::POOL_ERROR)
        return E_FAIL;
    }

    /* create session id */
    auto session_text = client_id + ":" + pool_name;
    session_id = std::to_string(CityHash32(session_text.c_str(), client_id.length()));
    PLOG("session id: (%s)", session_id.c_str());
    _open_pools[session_id] = Session{pool};
    _client_sessions[client_id].insert(session_id);
    return S_OK;
  }

  status_t close_pools(const std::string& client_id) {
    auto& session_set = _client_sessions[client_id];
    for(auto& session : session_set) {
      PLOG("closing pool Session (%s)", session.c_str());
      if(_open_pools.find(session) == _open_pools.end())
        throw General_exception("close pools cannot find session");
      close_pool(_open_pools[session].pool_handle);
      _open_pools.erase(session);
    }
    _client_sessions.erase(client_id);
    return S_OK;
  }

  inline status_t get_pool_names(std::list<std::string>& inout_pool_names) {
    return _itf->get_pool_names(inout_pool_names);
  }

  inline status_t put(const std::string& session_id, const std::string& key, const std::string& value) {
    
    if(_open_pools.find(session_id) == _open_pools.end()) {
      PWRN("cannot find pool for put command");
      return E_INVAL;
    }
    auto session = _open_pools[session_id];
    PLOG("put: (%s, %s)", key.c_str(), value.c_str());
    return _itf->put(session.pool_handle, key, value.data(), value.length());
  }
  
private:
  std::map<client_id_t, std::set<session_id_t>> _client_sessions;
  std::map<session_id_t, Session>               _open_pools;
};

class REST_endpoint : public Http::Endpoint
{
private:
  Rest::Router _router;
  bool         _use_ssl;
  Pool_manager _mgr;
  
  void get_status(const Rest::Request& request, Http::ResponseWriter response);
  void get_pools(const Rest::Request& request, Http::ResponseWriter response);
  void post_pools(const Rest::Request& request, Http::ResponseWriter response);
  void post_put(const Rest::Request& request, Http::ResponseWriter response);

public:
  explicit REST_endpoint(const Address addr,
                         const std::string& pmem_path,
                         const bool use_ssl,
                         const unsigned debug_level);

  ~REST_endpoint();

  void disconnect_hook(const std::string& client_id);

  void init(size_t thr = 2);
  
  void start(const std::string& server_cert_file,
             const std::string& server_key_file,
             const std::string& server_rootca_file);   

private:
  void setup_routes();
};

#pragma GCC diagnostic pop
#endif // __REST_ENDPOINT__
