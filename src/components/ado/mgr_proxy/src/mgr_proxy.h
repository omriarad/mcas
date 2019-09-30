#ifndef __ADOMGRPROX_COMPONENT_H__
#define __ADOMGRPROX_COMPONENT_H__

#include <api/ado_itf.h>
#include <common/queue.h>

// extern Common::Mpmc_bounded_lfq<Common::struct message> *queue;

class ADO_manager_proxy : public Component::IADO_manager_proxy {
public:
  ADO_manager_proxy(unsigned debug_level, int shard, std::string cores,
                    float cpu_num);
  ADO_manager_proxy();
  virtual ~ADO_manager_proxy();

  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0x8a120985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75,
                         0x21, 0xa1, 0x91); //

  void *query_interface(Component::uuid_t &itf_uuid) override {
    if (itf_uuid == Component::IADO_manager_proxy::iid()) {
      return (void *)static_cast<Component::IADO_manager_proxy *>(this);
    } else
      return NULL; // we don't support this interface
  }

  void unload() override { delete this; }

public:
  virtual Component::IADO_proxy *create(Component::IKVStore::pool_t pool_id,
                                        const std::string &filename,
                                        std::vector<std::string> &args,
                                        numa_node_t value_memory_numa_zone,
                                        Component::SLA *sla = nullptr) override;

  virtual bool has_exited(Component::IADO_proxy *ado_proxy) override;

  virtual status_t shutdown(Component::IADO_proxy *ado) override;

private:
  int shard;
  unsigned debug_level;
  shared_memory_token_t token;
  std::string cores;
  float cpu_num;
};

class ADO_manager_proxy_factory : public Component::IADO_manager_proxy_factory {
public:
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0xfac20985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75,
                         0x21, 0xa1, 0x91); //

  void *query_interface(Component::uuid_t &itf_uuid) override {
    if (itf_uuid == Component::IADO_manager_proxy_factory::iid()) {
      return (void *)static_cast<Component::IADO_manager_proxy_factory *>(this);
    } else
      return NULL; // we don't support this interface
  }

  void unload() override { delete this; }

  virtual Component::IADO_manager_proxy *create(unsigned debug_level, int core,
                                                std::string cores,
                                                float cpu_num) override {
    Component::IADO_manager_proxy *obj =
        static_cast<Component::IADO_manager_proxy *>(
            new ADO_manager_proxy(debug_level, core, cores, cpu_num));
    assert(obj);
    obj->add_ref();
    return obj;
  }
};

#endif
