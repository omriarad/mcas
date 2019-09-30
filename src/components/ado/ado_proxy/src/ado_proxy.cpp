#include "ado_proxy.h"
#include "ado_proto.h"
//#include
//"../../../../../../comanche/testing/kvstore/get_cpu_mask_from_string.h"
#include <common/logging.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sched.h>
#include <unistd.h>
#include <values.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric> /* accumulate */
#include <vector>

//#define USE_DOCKER
//#define USE_GDB /* WARNING: do not leave enabled; the regression tests won't work */

using namespace rapidjson;

static int proj_id = 1;

ADO_proxy::ADO_proxy(Component::IKVStore::pool_t pool_id,
                     const std::string &filename,
                     std::vector<std::string> &args, std::string cores,
                     int memory, float cpu_num, numa_node_t numa_zone)
    : _filename(filename), _args(args), _cores(cores), _memory(memory),
      _pool_id(pool_id), _core_number(cpu_num), _numa(numa_zone) {

  /* create unique channel id prefix */
  stringstream ss;
  ss << "channel-" << std::hex << (unsigned long)this;
  _channel_name = ss.str();

  _ipc = std::make_unique<ADO_protocol_builder>(
      _channel_name, ADO_protocol_builder::Role::CONNECT);

  // launch ado process
  this->launch();
}

ADO_proxy::~ADO_proxy() {
#ifdef USE_DOCKER
  docker_destroy(docker);
#endif
}

status_t ADO_proxy::bootstrap_ado() {
  _ipc->send_bootstrap();

  auto hr = _ipc->recv_bootstrap_response();
  if (hr == S_OK) {
    PMAJOR("ADO_proxy::bootstrap response OK.");
  } else {
    PWRN("ADO_proxy::invalid bootstrap response to ADO_proxy");
  }
  return hr;
}

status_t ADO_proxy::send_memory_map(uint64_t token, size_t size,
                                    void *value_vaddr) {
  PLOG("ADO_proxy: sending memory map request");
  _ipc->send_memory_map(token, size, value_vaddr);
  return S_OK;
}

status_t ADO_proxy::send_work_request(const uint64_t work_request_key,
                                      const std::string &work_key_str,
                                      const void *value_addr,
                                      const size_t value_len,
                                      const void *invocation_data,
                                      const size_t invocation_data_len) {
  _outstanding_wr++;

  _ipc->send_work_request(work_request_key, work_key_str, value_addr, value_len,
                          invocation_data, invocation_data_len);
  return S_OK;
}

void ADO_proxy::send_table_op_response(const status_t s, const void *value_addr,
                                       size_t value_len) {
  _ipc->send_table_op_response(s, value_addr, value_len);
}

bool ADO_proxy::check_work_completions(uint64_t& request_key,
                                       status_t& out_status,
                                       void *& out_response, /* use ::free to release */
                                       size_t & out_response_length)
{
  if(_outstanding_wr == 0) return false;
  
  auto result = _ipc->recv_from_ado_work_completion(request_key,
                                                    out_status,
                                                    out_response,
                                                    out_response_length);
  assert(_outstanding_wr != 0);

  if(result)
    _outstanding_wr--;
  
  return result;
}

bool ADO_proxy::check_table_ops(uint64_t &work_key,
                                int &op,
                                std::string &key,
                                size_t &value_len,
                                size_t &value_alignment,
                                void *& addr) {
  return _ipc->recv_table_op_request(work_key, op, key, value_len, value_alignment, addr);
}

void ADO_proxy::launch() {

#ifdef USE_DOCKER

  docker = docker_init(const_cast<char *>(std::string("v1.39").c_str()));
  if (!docker) {
    perror("Cannot initiate docker");
    return;
  }
  // create container
  Document req;
  req.SetObject();
  auto &allocator = req.GetAllocator();
  req.AddMember("Image",
                "res-mcas-docker-local.artifactory.swg-devops.com/ado:latest",
                allocator);
  Value config(kObjectType);
  config.AddMember("IpcMode", "host", allocator);
  config.AddMember("Privileged", true, allocator);
  Value vol(kArrayType);
  vol.PushBack("/tmp:/tmp", allocator);
  vol.PushBack("/dev:/dev", allocator);
  config.AddMember("Binds", vol.Move(), allocator);
  Value cap(kArrayType);
  cap.PushBack("ALL", allocator);
  config.AddMember("CapAdd", cap.Move(), allocator);
  config.AddMember("CpusetCpus", Value(_cores.c_str(), allocator).Move(),
                   allocator);
  config.AddMember("CpusetMems",
                   Value(to_string(_numa).c_str(), allocator).Move(),
                   allocator);
  config.AddMember("CpuPeriod", 100000, allocator);
  config.AddMember("CpuQuota", (int)(100000 * _core_number), allocator);
  //  Value env(kArrayType);
  // env.PushBack("LD_LIBRARY_PATH=/mcas/build/dist", allocator);
  // config.AddMember("Env", env.Move(), allocator);
  req.AddMember("HostConfig", config, allocator);
  Value cmd(kArrayType);
  _args.push_back("--channel_id");
  _args.push_back(_channel_name);
  for (auto &arg : _args) {
    cmd.PushBack(Value{}.SetString(arg.c_str(), arg.length(), allocator),
                 allocator);
  }
  req.AddMember("Cmd", cmd.Move(), allocator);

  StringBuffer sb;
  Writer<StringBuffer> writer(sb);
  req.Accept(writer);

  cout << sb.GetString() << endl;

  CURLcode response = docker_post(docker, "http://v1.39/containers/create",
                                  const_cast<char *>(sb.GetString()));
  assert(response == CURLE_OK);
  req.SetObject();
  req.Parse(docker_buffer(docker));
  container_id = req["Id"].GetString();

  // launch container
  string str("http://v1.39/containers/");
  str.append(container_id).append("/start");
  response = docker_post(docker, (char *)str.c_str(), NULL);
  assert(response == CURLE_OK);
#else

  /* run ADO process in GDB in an xterm window */
  stringstream cmd;

  /* run ADO process in an xterm window */
  std::vector<std::string> args{"/usr/bin/xterm", "-e"
#ifdef USE_GDB
    , "gdb", "--ex", "r", "--args"
#endif
      , _filename, "--channel", _channel_name, "--cpumask", _cores}; // TODO
  for (auto &arg : _args) { args.push_back(arg); }
  std::vector<char *> c;
  for ( auto &a : args )
  {
    c.push_back(&a[0]);
  }
  PLOG("cmd:%s"
    , std::accumulate(
        args.begin(), args.end()
        , std::string()
        , [](const std::string & a, const std::string & b)
          {
            return a + " " + b;
          }
      ).c_str()
    );
  c.push_back(nullptr);
  switch ( fork() )
  {
  case -1:
    throw General_exception("/usr/bin/xterm launch failed");
  case 0:
    throw General_exception("execv failed: %d", ::execv("/usr/bin/xterm", c.data()));
    break;
  default:
    break;
  }

  PLOG("ADO process launched: (%s)", _filename.c_str());

#endif
  _ipc->create_uipc_channels();

  return;
}

status_t ADO_proxy::kill() {

  _ipc->send_shutdown();
#ifdef USE_DOCKER
  string str("http://v1.39/containers/");
  str.append(container_id).append("/wait");
  CURLcode response = docker_post(docker, (char *)str.c_str(), NULL);
  assert(response == CURLE_OK);
  std::cout << docker_buffer(docker) << std::endl;
  str.clear();

  //  msgctl(_msqid, IPC_RMID, 0);
  // remove container
  str.append("http://v1.39/containers/").append(container_id);
  response = docker_delete(docker, (char *)str.c_str(), NULL);
  assert(response == CURLE_OK);
#else
  /* should be gracefully closed */
#endif
  return S_OK;
}

bool ADO_proxy::has_exited() { return false; }

status_t ADO_proxy::shutdown() { return kill(); }

void ADO_proxy::add_deferred_unlock(const uint64_t work_key,
                                    const Component::IKVStore::key_t key) {
  _deferred_unlocks[work_key].push_back(key);
}

void ADO_proxy::get_deferred_unlocks(
    const uint64_t work_key, std::vector<Component::IKVStore::key_t> &keys) {
  auto &v = _deferred_unlocks[work_key];
  keys.assign(v.begin(), v.end());
  v.clear();
}

/**
 * Factory entry point
 *
 */
extern "C" void *factory_createInstance(Component::uuid_t &component_id) {
  if (component_id == ADO_proxy_factory::component_id()) {
    return static_cast<void *>(new ADO_proxy_factory());
  } else
    return NULL;
}
