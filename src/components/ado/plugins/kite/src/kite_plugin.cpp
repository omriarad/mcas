#include "kite_plugin.h"
#include <api/interfaces.h>
#include <common/logging.h>
//#include <nupm/pm_lowlevel.h>
#include <string>

using namespace std;

status_t ADO_kite_plugin::register_mapped_memory(void *shard_vaddr,
                                                 void *local_vaddr,
                                                 size_t len) {
  PLOG("ADO_kite_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr,
       local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

status_t ADO_kite_plugin::shutdown() {
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

status_t ADO_kite_plugin::do_work(
    const uint64_t work_key, const std::string &key, void *shard_value_vaddr,
    size_t value_len,
    const void *in_work_request, /* don't use iovec because of non-const */
    const size_t in_work_request_len, void *&out_work_response,
    size_t &out_work_response_len) {
  PLOG("ADO_kite_plugin: work_id (%lu)", work_key);
  PLOG("ADO_kite_plugin: do_work (%s, value_addr=%p, valuen_len=%lu)",
       key.c_str(), shard_value_vaddr, value_len);
  PLOG("Current value: %.*s", (int)value_len, (char *)shard_value_vaddr);
  void *old_value_addr = nullptr;
  size_t old_value_len;
  _cb.open_key_func(work_key, key, old_value_addr, old_value_len);
  uint64_t new_value = stoull(static_cast<const char *>(in_work_request));

  out_work_response_len = 3;
  out_work_response = ::malloc(out_work_response_len);
  strncpy((char *)out_work_response, "OK!", 3);

  if (old_value_addr) {
    unsigned i = 0;
    uint64_t old_value = *((uint64_t *)old_value_addr + i);
    while (old_value != MAGIC_NUMBER && i < old_value_len / 8) {
      if (old_value == new_value) {
        // just return, no need to do anything
        PMAJOR("duplicated gnome id for kmer");
        return S_OK;
      }
      i++;
      old_value = *((uint64_t *)old_value_addr + i);
    }
    // find the index of the end, and no duplicated ids
    if (i == old_value_len / 8) {
      // TODO:need to resize the value, how?
    }
    *((uint64_t *)old_value_addr + i) = new_value;
    *((uint64_t *)old_value_addr + i + 1) = MAGIC_NUMBER;
    PMAJOR("attached new gnome id");
    return S_OK;
  }

  void *new_value_addr = nullptr;
  _cb.create_key_func(work_key, key, VALUE_SIZE,
                      new_value_addr); // TODO: what size?
  *((uint64_t *)new_value_addr) = new_value;
  *((uint64_t *)new_value_addr + 1) = MAGIC_NUMBER;
  PMAJOR("created new kmer");
  return S_OK;

  // memset(shard_value_vaddr, 'X', value_len);
  // nupm::mem_flush(shard_value_vaddr, value_len);
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(Component::uuid_t &interface_iid) {
  PLOG("instantiating ADO_kite_plugin");
  if (interface_iid == Interface::ado_plugin)
    return static_cast<void *>(new ADO_kite_plugin());
  else
    return NULL;
}
