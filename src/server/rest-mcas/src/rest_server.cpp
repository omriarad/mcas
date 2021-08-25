#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <pistache/endpoint.h>
#include <pistache/http.h>
#include <pistache/router.h>
#pragma GCC diagnostic pop

#include <boost/program_options.hpp>
#include <iostream>
#include "rest_server_config.h"
#include "endpoint.h"

using namespace Pistache;

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
  po::variables_map vm;
  
  try {
    po::options_description desc("Options");
    desc.add_options()
      ("help", "Show help")
      ("ssl","Use SSL connection (TLS)")
      ("certfile", po::value<std::string>()->default_value(REST_MCAS_SOURCE_DIR "certs/server/server.crt"), "Server certificate file")
      ("keyfile", po::value<std::string>()->default_value(REST_MCAS_SOURCE_DIR "certs/server/server.key"), "Server private key file")
      ("rootcafile", po::value<std::string>()->default_value(REST_MCAS_SOURCE_DIR "certs/rootCA/rootCA.crt"), "Server root CA file")
      ("pmem", po::value<std::string>()->default_value("/dev/dax0.0"), "PMEM path")
      ("threads", po::value<unsigned>()->default_value(2), "Thread count")
      ("port", po::value<uint16_t>()->default_value(9999), "Network port")
      ;

    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }
  }
  catch (...) {
    std::cerr << "bad command line option configuration\n";
    return -1;
  }
  
  auto threads = vm["threads"].as<unsigned>();
  auto pmem = vm["pmem"].as<std::string>();
  auto opts = Http::Endpoint::options().threads(threads);
  auto port = vm["port"].as<uint16_t>();
  auto certfile = vm["certfile"].as<std::string>();
  auto keyfile = vm["keyfile"].as<std::string>();
  auto rootcafile = vm["rootcafile"].as<std::string>();
  Pistache::Address addr = Pistache::Address(Pistache::Ipv4::any(), Pistache::Port(port));

  std::cout << "Starting MCAS REST server\n";
  std::cout << "Port:" << port << "\n";
  std::cout << "Threads:" << threads << "\n";

  REST_endpoint server(addr, pmem, vm.count("ssl"));  
  server.init(threads);
  server.start(certfile, keyfile, rootcafile);
                     
  return 0;
}
