#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <pistache/endpoint.h>
#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/peer.h>
#include <pistache/serializer/rapidjson.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include <vector>
#include <mutex>

#include <common/logging.h>
#include <common/str_utils.h>


#include "endpoint.h"

using namespace rapidjson;

static constexpr size_t         DEFAULT_POOL_SIZE_MB = 32;

static std::mutex               g_lock_endpoint_table;
static std::set<REST_endpoint*> g_endpoint_table;



static void print_cookies(const Http::Request& req)
{
  auto cookies = req.cookies();
  std::cout << "Cookies: [" << std::endl;
  const std::string indent(4, ' ');
  for (const auto& c : cookies)
    {
      std::cout << indent << c.name << " = " << c.value << std::endl;
    }
  std::cout << "]" << std::endl;
}

/* we have to do this because the disconnect handler does support
   passing of the self REST_endpoint */
static void global_add_endpoint(REST_endpoint * ep_ptr) {
  std::lock_guard<std::mutex> lock(g_lock_endpoint_table);
  g_endpoint_table.insert(ep_ptr);
}

static void global_remove_endpoint(REST_endpoint * ep_ptr) {
  std::lock_guard<std::mutex> lock(g_lock_endpoint_table);
  auto i = g_endpoint_table.find(ep_ptr);
  assert(i != g_endpoint_table.end());
  g_endpoint_table.erase(i);
}

static void global_call_disconnect(const std::shared_ptr<Tcp::Peer>& peer) {
  std::lock_guard<std::mutex> lock(g_lock_endpoint_table);
  
  auto addr = peer.get()->address();
  std::stringstream ss;
  ss << addr.host() << ":" << static_cast<uint16_t>(addr.port());

  for(auto& e : g_endpoint_table) {
    e->disconnect_hook(ss.str());
  }
}


// static void disconnectHandler(const std::shared_ptr<Tcp::Peer>& peer)
// {
//   auto addr = peer.get()->address();
//   std::stringstream ss;
//   ss << addr.host() << ":" << static_cast<uint16_t>(addr.port());
  
//   PNOTICE("DISCONNECT FROM CLIENT (%s)", ss.str().c_str());
// }

static std::string client_host_id(const Rest::Request& request)
{
  auto addr = request.address();
  std::stringstream ss;
  ss << addr.host() << ":" << static_cast<uint16_t>(addr.port());
  return ss.str();
}



using namespace Pistache;



REST_endpoint::REST_endpoint(const Address addr,
                             const std::string& pmem_path,
                             const bool use_ssl,
                             const unsigned debug_level)
  : Http::Endpoint(addr),
    _router(),
    _use_ssl(use_ssl),
    _mgr(pmem_path, debug_level)
{
  global_add_endpoint(this);
}

REST_endpoint::~REST_endpoint()
{
  global_remove_endpoint(this);
}

void REST_endpoint::init(size_t thr)
{
  Options opts = Http::Endpoint::options().threads(static_cast<int>(thr));
  Http::Endpoint::init(opts);
  setup_routes();
}

void REST_endpoint::start(const std::string& server_cert_file,
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



void REST_endpoint::setup_routes()
{
  using namespace Rest;

  /* Example URL: /pools/myPool?sizemb=128 */
  Routes::Post(_router, "/pools/:name", Routes::bind(&REST_endpoint::post_pools, this));

  /* Example URL: /put?pool=24344354?key=foo?value=bar */
  Routes::Post(_router, "/put", Routes::bind(&REST_endpoint::post_put, this));

  /* Example URL: /pools */
  Routes::Get(_router, "/pools", Routes::bind(&REST_endpoint::get_pools, this));

  _router.addDisconnectHandler(global_call_disconnect);
}


void REST_endpoint::get_status(const Rest::Request& request,
                               Http::ResponseWriter response)
{
  response.send(Http::Code::Ok, "OK!\n");
}


void REST_endpoint::get_pools(const Rest::Request& request, Http::ResponseWriter response)
{
  /* look for pools?type=xxx param */
  // if (request.query().has("type")) {
  //   PLOG("has type param: %s", request.query().get("type")->c_str());
  // }
  
  Document doc;
  Document::AllocatorType& allocator = doc.GetAllocator();
  
  doc.SetArray();

  std::list<std::string> names;
  if(_mgr.get_pool_names(names) != S_OK)
    throw Logic_exception("IKVStire::get_pool_names invocation failed unexpectedly");



  for(auto& name : names) {
    doc.PushBack(Value().SetString(name.c_str(),boost::numeric_cast<rapidjson::SizeType>(name.length())), allocator);
    //doc.PushBack(Value().SetStringRaw(name.c_str()), allocator);
    //GenericStringRef s(name.c_str(), allocator);
    //    doc.AddMember("foo","nar",allocator);
  }
  
  StringBuffer sb;
  Writer<StringBuffer, Document::EncodingType, ASCII<> > writer(sb);
  doc.Accept(writer);

  response.send(Http::Code::Ok, sb.GetString()); //"Pools OK!\n");
}

  
void REST_endpoint::post_pools(const Rest::Request& request, Http::ResponseWriter response)
{
  auto pool_name = request.param(":name").as<std::string>();
  PLOG("post_pool: (%p,req=%p) name=%s", reinterpret_cast<void*>(this), reinterpret_cast<const void*>(&request), pool_name.c_str());

  PLOG("post_pool: client (%s)",client_host_id(request).c_str());

  size_t size_mb = DEFAULT_POOL_SIZE_MB;
  if (request.query().has("sizemb")) {
    size_mb = std::stoul(*request.query().get("sizemb"));
  }

  session_id_t session_cookie;
  if(_mgr.create_or_open_pool(client_host_id(request), pool_name, size_mb, session_cookie) == S_OK) {

      Document doc;
      Document::AllocatorType& allocator = doc.GetAllocator();

      doc.SetObject();
      doc.AddMember("session", session_cookie, allocator);
      // Value result;
      // result.AddMember("name", "Milo", doc.GetAllocator());
      // doc.SetArray();
      // doc.PushBack(Value().SetString("hstore"), allocator);
  
      StringBuffer sb;
      Writer<StringBuffer, Document::EncodingType, ASCII<> > writer(sb);
      doc.Accept(writer);

      response.send(Http::Code::Ok, sb.GetString()); //"Pools OK!\n");     
      return;
  }

  response.send(Http::Code::Bad_Request, "{\"status\" : -1}");
}
                                
void REST_endpoint::disconnect_hook(const std::string& client_id) {
  PNOTICE("--> disconnect (%s)", client_id.c_str());
  _mgr.close_pools(client_id);
}

void REST_endpoint::post_put(const Rest::Request& request, Http::ResponseWriter response)
{
  if (!request.query().has("pool") ||
      !request.query().has("key") ||
      !request.query().has("value")) {
    
    response.send(Http::Code::Bad_Request, "{\"status\" : -1}");
    return;
  }
  
  std::string pool = *request.query().get("pool");
  std::string key = *request.query().get("key");
  std::string value = *request.query().get("value");
  
  _mgr.put(pool, key, value);

  response.send(Http::Code::Ok, "{\"status\" : 0}");
  return;
}

#pragma GCC diagnostic pop
