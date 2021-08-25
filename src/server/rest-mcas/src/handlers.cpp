#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wpedantic"

#include <pistache/endpoint.h>
#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/serializer/rapidjson.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/document.h>

#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include <common/logging.h>
#include <common/str_utils.h>

#include "endpoint.h"

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

namespace Generic
{

void handle_ready(const Rest::Request&, Http::ResponseWriter response)
{
  response.send(Http::Code::Ok, "OK!\n");
}

}

 
void REST_endpoint::get_status(const Rest::Request& request,
                               Http::ResponseWriter response)
{
  response.send(Http::Code::Ok, "OK!\n");
}


void REST_endpoint::get_pools(const Rest::Request& request, Http::ResponseWriter response)
{
  using namespace rapidjson;

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


void REST_endpoint::post_pool(const Rest::Request& request, Http::ResponseWriter response)
{
  auto name = request.param(":name").as<std::string>();
  PLOG("post_pool: (%p,req=%p) name=%s", reinterpret_cast<void*>(this), reinterpret_cast<const void*>(&request), name.c_str());
  //  if (request.hasParam(":name"))

  response.send(Http::Code::Ok, "OK!\n");
}
                                


#pragma GCC diagnostic pop
