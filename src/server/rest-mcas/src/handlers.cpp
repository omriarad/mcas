#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <pistache/endpoint.h>
#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/serializer/rapidjson.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include <common/logging.h>
#include <common/str_utils.h>

#include "endpoint.h"

using namespace rapidjson;

static constexpr size_t DEFAULT_POOL_SIZE_MB = 32;

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

using namespace Pistache;

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
  doc.PushBack(Value().SetString("hstore"), allocator);
  
  StringBuffer sb;
  Writer<StringBuffer, Document::EncodingType, ASCII<> > writer(sb);
  doc.Accept(writer);

  response.send(Http::Code::Ok, sb.GetString()); //"Pools OK!\n");
}

static std::string client_host_id(const Rest::Request& request)
{
  const Pistache::Address addr = request.address();
  std::stringstream ss;
  ss << addr.host() << ":" << static_cast<uint16_t>(addr.port());
  return ss.str();
}
  
void REST_endpoint::post_pools(const Rest::Request& request, Http::ResponseWriter response)
{
  auto pool_name = request.param(":name").as<std::string>();
  PLOG("post_pool: (%p,req=%p) name=%s", reinterpret_cast<void*>(this), reinterpret_cast<const void*>(&request), pool_name.c_str());

  PLOG("post_pool: client (%s)",client_host_id(request).c_str());

  size_t size_mb = DEFAULT_POOL_SIZE_MB;
  if (request.query().has("sizemb")) {
    size_mb = std::stoul(*request.query().get("sizemb"));
    PLOG("Has size! (%lu)", size_mb);
  }

  session_id session_cookie;
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
                                


#pragma GCC diagnostic pop
