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

#include "rest_server_config.h"

using namespace Pistache;

namespace Generic
{
void handle_ready(const Rest::Request&, Http::ResponseWriter response);
}


class REST_endpoint : public Http::Endpoint
{
private:
  Rest::Router _router;

  void status(const Rest::Request& request, Http::ResponseWriter response);
  void get_pools(const Rest::Request& request, Http::ResponseWriter response);

public:
  explicit REST_endpoint(Address addr, bool use_ssl = false)
    : Http::Endpoint(addr), _router()
  {
    if(use_ssl) {
      useSSL(REST_MCAS_SOURCE_DIR "certs/server/server.crt",
             REST_MCAS_SOURCE_DIR "certs/server/server.key");
      useSSLAuth(REST_MCAS_SOURCE_DIR "certs/rootCA/rootCA.crt");
      std::cout << "SSL: enabled\n";
    }
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
    Http::Endpoint::serve();
  }

  void setup_routes()
  {
    using namespace Rest;

    // Routes::Post(router, "/record/:name/:value?", Routes::bind(&StatsEndpoint::doRecordMetric, this));
    // Routes::Get(router, "/value/:name", Routes::bind(&StatsEndpoint::doGetMetric, this));
    // Routes::Get(router, "/ready", Routes::bind(&Generic::handleReady));
    Routes::Get(_router, "/status", Routes::bind(&REST_endpoint::status, this));
    Routes::Get(_router, "/pools", Routes::bind(&REST_endpoint::get_pools, this));
  }

};

#endif // __REST_ENDPOINT__
