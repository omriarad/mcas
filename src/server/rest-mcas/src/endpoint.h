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


#include <api/components.h>
#include <api/kvstore_itf.h>

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
          const std::string& store = "hstore-cc",
          const unsigned debug_level = 3) {
    using namespace component;

    std::string store_lib = "libcomponent-" + store + ".so";
    IBase * comp = load_component(store_lib.c_str(), component::hstore_factory);
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
  
private:
  static constexpr uint64_t               _load_addr = 0x8800000000;
  component::Itf_ref<component::IKVStore> _itf;
};

class REST_endpoint : public Http::Endpoint
{
private:
  Rest::Router _router;
  bool         _use_ssl;
  Backend      _store;
  
  void get_status(const Rest::Request& request, Http::ResponseWriter response);
  void get_pools(const Rest::Request& request, Http::ResponseWriter response);
  void post_pool(const Rest::Request& request, Http::ResponseWriter response);

public:
  explicit REST_endpoint(Address addr,
                         const std::string& pmem_path,
                         bool use_ssl = false)
    : Http::Endpoint(addr),
      _router(),
      _use_ssl(use_ssl),
      _store(pmem_path)
  {
  }

  void init(size_t thr = 2)
  {
    auto opts = Http::Endpoint::options().threads(static_cast<int>(thr));
    Http::Endpoint::init(opts);
    setup_routes();
  }

  void start()
  {
    Http::Endpoint::setHandler(_router.handler());
    
    if(_use_ssl) {
      useSSL(REST_MCAS_SOURCE_DIR "certs/server/server.crt",
             REST_MCAS_SOURCE_DIR "certs/server/server.key");
      useSSLAuth(REST_MCAS_SOURCE_DIR "certs/rootCA/rootCA.crt");
      std::cout << "SSL: enabled\n";
    }

    Http::Endpoint::serve();
  }

  void setup_routes()
  {
    using namespace Rest;

    // Routes::Post(router, "/record/:name/:value?", Routes::bind(&StatsEndpoint::doRecordMetric, this));
    // Routes::Get(router, "/value/:name", Routes::bind(&StatsEndpoint::doGetMetric, this));
    // Routes::Get(router, "/ready", Routes::bind(&Generic::handleReady));
    Routes::Get(_router, "/status", Routes::bind(&REST_endpoint::get_status, this));
    Routes::Get(_router, "/pools", Routes::bind(&REST_endpoint::get_pools, this));
    Routes::Post(_router, "/pool/:name", Routes::bind(&REST_endpoint::post_pool, this));
  }

};

#pragma GCC diagnostic pop
#endif // __REST_ENDPOINT__
