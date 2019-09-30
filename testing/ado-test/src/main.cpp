#include <stdio.h>
#include <string>
#include <sstream>
#include <chrono>
#include <iostream>
#include <common/str_utils.h>
#include <common/cycles.h>
#include <boost/program_options.hpp>
#include <api/components.h>
#include <api/mcas_itf.h>

struct Options
{
  unsigned debug_level;
  std::string server;
  std::string device;
  unsigned port;
  bool async;
} g_options;

Component::IMCAS* init(const std::string& server_hostname,  int port);
void test0(Component::IMCAS* mcas);
void test1(Component::IMCAS* mcas);
void test2(Component::IMCAS* mcas); /* structured test */

int main(int argc, char * argv[])
{
  namespace po = boost::program_options;

  try {
    po::options_description desc("Options");

    desc.add_options()
      ("help", "Show help")
      ("server", po::value<std::string>()->default_value("10.0.0.21"), "Server hostname")
      ("device", po::value<std::string>()->default_value("mlx5_0"), "Device (e.g. mlnx5_0)")
      ("port", po::value<unsigned>()->default_value(11911), "Server port")
      ("debug", po::value<unsigned>()->default_value(0), "Debug level")
      ("test", po::value<unsigned>()->default_value(0), "Test number")
      ("async", "Use asynchronous invocation")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    if (vm.count("server") == 0) {
      std::cout << "--server option is required\n";
      return -1;
    }

    g_options.server = vm["server"].as<std::string>();
    g_options.device = vm["device"].as<std::string>();
    g_options.port = vm["port"].as<unsigned>();
    g_options.debug_level = vm["debug"].as<unsigned>();
    g_options.async = vm.count("async");
    
    // mcas::Global::debug_level = g_options.debug_level =
    //     vm["debug"].as<unsigned>();
    auto mcasptr = init(vm["server"].as<std::string>(), vm["port"].as<unsigned>());

    switch(vm["test"].as<unsigned>()) {
    case 0:
      test0(mcasptr);
      break;
    case 1:
      test1(mcasptr);
      break;
    case 2:
      test2(mcasptr);
      break;      
    default:
      PLOG("invalid test selected");
    }
    mcasptr->release_ref();
  }
  catch (po::error e) {
    printf("bad command line option\n");
    return -1;
  }
 
  return 0;
}

Component::IMCAS * init(const std::string& server_hostname,  int port)
{
  using namespace Component;
  
  IBase *comp = Component::load_component("libcomponent-mcasclient.so",
                                          mcas_client_factory);

  auto fact = (IMCAS_factory *) comp->query_interface(IMCAS_factory::iid());
  if(!fact)
    throw Logic_exception("unable to create MCAS factory");

  std::stringstream url;
  url << g_options.server << ":" << g_options.port;
  
  IMCAS * mcas = fact->mcas_create(g_options.debug_level,
                                   "None",
                                   url.str(),
                                   g_options.device);

  if(!mcas)
    throw Logic_exception("unable to create MCAS client instance");

  fact->release_ref();

  return mcas;
}

void test0(Component::IMCAS* mcas)
{
  using namespace Component;

  const std::string poolname = "pool0";
  
  auto pool = mcas->create_pool(poolname,
                                64, /* size */
                                0, /* flags */
                                100); /* obj count */
  if(pool == IKVStore::POOL_ERROR)
    throw General_exception("create_pool (%s) failed", poolname.c_str());


  mcas->erase(pool, "test0");

  std::string response;
  status_t rc;
  rc = mcas->invoke_ado(pool,
                        "test0",
                        "RUN!TEST-0",
                        IMCAS::ADO_FLAG_CREATE_ON_DEMAND,
                        response,
                        KB(4));
  PLOG("test0: rc=%d", rc);
  mcas->close_pool(pool);
}

void test1(Component::IMCAS* mcas)
{
  using namespace Component;

  const std::string poolname = "pool0";
  
  auto pool = mcas->create_pool(poolname,
                                64, /* size */
                                0, /* flags */
                                100); /* obj count */
  if(pool == IKVStore::POOL_ERROR)
    throw General_exception("create_pool (%s) failed", poolname.c_str());


  mcas->erase(pool, "test1");

  std::string response;
  status_t rc;
  std::string cmd = "RUN!TEST-1";
  std::string value_to_put = "VALUE_TO_PUT";
  rc = mcas->invoke_put_ado(pool,
                            "test1",
                            cmd.data(),
                            cmd.length(),
                            value_to_put.data(),
                            value_to_put.length(),
                            IMCAS::ADO_FLAG_CREATE_ON_DEMAND,
                            response);
  assert(rc == S_OK);

  {
    void * ptr=nullptr;
    size_t ptr_len = 0;
    rc = mcas->get(pool, "test1", ptr, ptr_len);
    std::string r((char *) ptr, ptr_len);
    PLOG("got back: %s", r.c_str());
    assert(r == value_to_put);
  }
  
  PLOG("test1: rc=%d", rc);
  mcas->close_pool(pool);
}


void test2(Component::IMCAS* mcas)
{
  using namespace Component;

  PLOG("running test2...");
  
  const std::string poolname = "pool0";
  
  auto pool = mcas->create_pool(poolname,
                                64, /* size */
                                0, /* flags */
                                100); /* obj count */
  if(pool == IKVStore::POOL_ERROR)
    throw General_exception("create_pool (%s) failed", poolname.c_str());

  status_t rc;
  std::string response;
  rc = mcas->invoke_ado(pool,
                        "test2",
                        "RUN!TEST-2",
                        IMCAS::ADO_FLAG_CREATE_ON_DEMAND,
                        response,
                        KB(4));

  PINF("test2: %d [%s]", rc, response.c_str());
  mcas->close_pool(pool);
}
