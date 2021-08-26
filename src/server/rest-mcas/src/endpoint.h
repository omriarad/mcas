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
          const std::string& store = "mapstore",
          const unsigned debug_level = 3) {
    using namespace component;

    std::string store_lib = "libcomponent-" + store + ".so";
    IBase * comp;


    if(store == "hstore-cc") { /* HSTORE-CC */
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

  auto kvstore() {
    return _itf.get();
  }
  
private:
  static constexpr uint64_t               _load_addr = 0x8800000000;

protected:
  component::Itf_ref<component::IKVStore> _itf;
};

using session_id = std::string;

class Pool_manager : private Backend
{
public:
  
  struct Session {
    component::IKVStore::pool_t pool_handle;
  };

  Pool_manager(const std::string& pmem_path) : Backend(pmem_path) {
  }

  status_t create_or_open_pool(const std::string& client_id, const std::string& pool_name, const size_t size_mb, session_id& session) {
    auto pool = _itf->open_pool(pool_name);
    if(pool != component::IKVStore::POOL_ERROR) {
      PLOG("endpoint: opened existing pool");
    }
    else {
      pool = _itf->create_pool(pool_name, MiB(size_mb));
      if(pool == component::IKVStore::POOL_ERROR)
        return E_FAIL;
    }
    /* create session id */
    auto session_text = client_id + ":" + pool_name;
    session = std::to_string(CityHash32(session_text.c_str(), client_id.length()));
    _open_pools[session] = Session{pool};
    return S_OK;
  }
  
private:
  std::map<session_id, Session> _open_pools;
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

public:
  explicit REST_endpoint(Address addr,
                         const std::string& pmem_path,
                         bool use_ssl = false)
    : Http::Endpoint(addr),
      _router(),
      _use_ssl(use_ssl),
      _mgr(pmem_path)
  {
  }

  void init(size_t thr = 2)
  {
    auto opts = Http::Endpoint::options().threads(static_cast<int>(thr));
    Http::Endpoint::init(opts);
    setup_routes();
  }

  void start(const std::string& server_cert_file,
             const std::string& server_key_file,
             const std::string& server_rootca_file)
  {
    Http::Endpoint::setHandler(_router.handler());
    
    if(_use_ssl) {
      useSSL(server_cert_file, server_key_file);
      useSSLAuth(server_rootca_file);
      std::cout << "SSL: enabled\n";
    }

    Http::Endpoint::serve();
  }

  void setup_routes()
  {
    using namespace Rest;

    /* Example URL: /pools/myPool?sizemb=128 */
    Routes::Post(_router, "/pools/:name", Routes::bind(&REST_endpoint::post_pools, this));

    /* Example URL: /pools */
    Routes::Get(_router, "/pools", Routes::bind(&REST_endpoint::get_pools, this));
    
    // Routes::Get(router, "/value/:name", Routes::bind(&StatsEndpoint::doGetMetric, this));
    // Routes::Get(router, "/ready", Routes::bind(&Generic::handleReady));
    //    Routes::Get(_router, "/status", Routes::bind(&REST_endpoint::get_status, this));

    //    Routes::Post(_router, "/pool/:name", Routes::bind(&REST_endpoint::post_pool, this));
  }

};

#pragma GCC diagnostic pop
#endif // __REST_ENDPOINT__
