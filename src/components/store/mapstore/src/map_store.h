/*
  Copyright [2017-2020] [IBM Corporation]
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __MAP_STORE_COMPONENT_H__
#define __MAP_STORE_COMPONENT_H__

#include <api/kvstore_itf.h>
#include "map_store_env.h"

#define PREFIX "Map_store: "

namespace nupm
{
  struct region_descriptor;
}

class Map_store : public component::IKVStore /* generic Key-Value store interface */
{
  unsigned          _debug_level;
  const std::string _mm_plugin_path;
public:
  
  unsigned debug_level() { return _debug_level; }

  /**
   * Constructor
   *
   * @param block_device Block device interface
   *
   */
  Map_store(const unsigned debug_level,
            const std::string &mm_plugin_path,
            const std::string &owner,
            const std::string &name);

  /**
   * Destructor
   *
   */
  ~Map_store();

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0x8a120985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75,
                         0x21, 0xa1, 0x21);

  void *query_interface(component::uuid_t &itf_uuid) override {
    if (itf_uuid == component::IKVStore::iid()) {
      return static_cast<component::IKVStore *>(this);
    }
    else {
      return NULL;  // we don't support this interface
    }
  }

  void unload() override { delete this; }

public:
  /* IKVStore */
  virtual int thread_safety() const override { return THREAD_MODEL_RWLOCK_PER_POOL; }

  virtual int get_capability(Capability cap) const override;

  virtual pool_t create_pool(const std::string &name, const size_t size,
                             flags_t flags = 0,
                             uint64_t expected_obj_count = 0,
                             component::IKVStore::Addr base_addr_unused = component::IKVStore::Addr{0}) override;

  virtual pool_t open_pool(const std::string &name,
                           flags_t flags = 0,
                           component::IKVStore::Addr base_addr_unused = component::IKVStore::Addr{0}) override;

  virtual status_t close_pool(const pool_t pid) override;

  virtual status_t delete_pool(const std::string &name) override;

  virtual status_t put(const pool_t pool, const std::string &key,
                       const void *value, const size_t value_len,
                       flags_t flags = FLAGS_NONE) override;

  virtual status_t get(const pool_t pool, const std::string &key,
                       void *&out_value, size_t &out_value_len) override;

  virtual status_t get_direct(const pool_t pool, const std::string &key, void *out_value,
                              size_t &out_value_len,
                              IKVStore::memory_handle_t handle) override;

  virtual status_t put_direct(const pool_t pool, const std::string &key,
                              const void *value, const size_t value_len,
                              IKVStore::memory_handle_t handle = HANDLE_NONE,
                              flags_t flags = FLAGS_NONE) override;

  virtual status_t resize_value(const pool_t pool, const std::string &key,
                                const size_t new_size,
                                const size_t alignment) override;

  virtual status_t get_attribute(const pool_t pool, const Attribute attr,
                                 std::vector<uint64_t> &out_attr,
                                 const std::string *key = nullptr) override;

  virtual status_t swap_keys(const pool_t pool,
                             const std::string key0,
                             const std::string key1) override;

  virtual status_t lock(const pool_t pool,
                        const std::string &key,
                        lock_type_t type,
                        void *&out_value,
                        size_t &inout_value_len,
                        size_t alignment,
                        IKVStore::key_t &out_key,
                        const char ** out_key_ptr) override;

  virtual status_t unlock(const pool_t pool,
                          key_t key,
                          IKVStore::unlock_flags_t flags) override;

  virtual status_t erase(const pool_t pool, const std::string &key) override;

  virtual size_t count(const pool_t pool) override;

  virtual status_t free_memory(void *p) override;

  virtual status_t map(const pool_t pool,
                       std::function<int(const void * key,
                                         const size_t key_len,
                                         const void *value,
                                         const size_t value_len)> function) override;

  virtual status_t map(const pool_t pool,
                       std::function<int(const void* key,
                                         const size_t key_len,
                                         const void* value,
                                         const size_t value_len,
                                         const common::tsc_time_t timestamp)> function,
                       const common::epoch_time_t t_begin,
                       const common::epoch_time_t t_end) override;

  virtual status_t map_keys(const pool_t pool,
                            std::function<int(const std::string &key)> function) override;

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;
  
  virtual status_t get_pool_regions(const pool_t pool,
                                    nupm::region_descriptor &out_regions) override;
  
  virtual status_t grow_pool(const pool_t pool, const size_t increment_size,
                             size_t &reconfigured_size) override;
  
  virtual status_t free_pool_memory(const pool_t pool, const void *addr,
                                    const size_t size = 0) override;
  
  virtual status_t allocate_pool_memory(const pool_t pool, const size_t size,
                                        const size_t alignment,
                                        void *&out_addr) override;

  virtual IKVStore::pool_iterator_t open_pool_iterator(const pool_t pool) override;

  virtual status_t deref_pool_iterator(const pool_t pool,
                                       pool_iterator_t iter,
                                       const common::epoch_time_t t_begin,
                                       const common::epoch_time_t t_end,
                                       pool_reference_t& ref,
                                       bool& time_match,
                                       bool increment = true) override;

  virtual status_t close_pool_iterator(const pool_t pool,
                                       pool_iterator_t iter) override;
};

class Map_store_factory : public component::IKVStore_factory {
public:
  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(1.0f);
  DECLARE_COMPONENT_UUID(0xfac20985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75, 0x21, 0xa1, 0x21);

  virtual ~Map_store_factory() {
  }
  
  void *query_interface(component::uuid_t &itf_uuid) override {
    if (itf_uuid == component::IKVStore_factory::iid()) {
      return static_cast<component::IKVStore_factory *>(this);
    }
    else return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  virtual component::IKVStore *create(unsigned debug_level,
                                      const IKVStore_factory::map_create &mc) override
  {
    auto owner_it = mc.find(+component::IKVStore_factory::k_owner);
    auto name_it = mc.find(+component::IKVStore_factory::k_name);
    auto mm_plugin_path_it = mc.find(+component::IKVStore_factory::k_mm_plugin_path);

    std::string checked_mm_plugin_path;
    if(mm_plugin_path_it == mc.end()) {
      checked_mm_plugin_path = DEFAULT_MM_PLUGIN_PATH;
    }
    else {
      std::string path = mm_plugin_path_it->second;

      if(access(path.c_str(), F_OK) != 0) {
        path = DEFAULT_MM_PLUGIN_LOCATION + path;
        if(access(path.c_str(), F_OK) != 0) {
          PERR("inaccessible plugin path (%s) and (%s)", mm_plugin_path_it->second.c_str(), path.c_str());
          throw General_exception("unable to open mm_plugin");
        }
        checked_mm_plugin_path = path;
      }
      else {
        checked_mm_plugin_path = path;
      }
    }
    
    component::IKVStore *obj =
      static_cast<component::IKVStore *>
      (new Map_store(debug_level,
                     checked_mm_plugin_path,
                     owner_it == mc.end() ? "owner" : owner_it->second,
                     name_it == mc.end() ? "name" : name_it->second));
    assert(obj);
    obj->add_ref();
    return obj;
  }
};

#endif
