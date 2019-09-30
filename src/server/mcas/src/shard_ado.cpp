/*
  Copyright [2017-2019] [IBM Corporation]
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

/*< #included in shard.cpp */
#include <api/ado_itf.h>
#include <nupm/mcas_mod.h>
#include <cstdint> /* PRIu64 */

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

//#define SHORT_CIRCUIT_ADO_HANDLING

status_t Shard::conditional_bootstrap_ado_process(Connection_handler* handler,
                                                  Component::IKVStore::pool_t pool_id,
                                                  Component::IADO_proxy *& ado)
{
  /* ADO processes are instantiated on a per-pool basis.  First
     check if an ADO process already exists.
  */
  bool bootstrap = true;

  auto i = _ado_map.find(pool_id);

  if(i == _ado_map.end()) {
    /* need to launch new ADO process */
    std::vector<std::string> args;
    args.push_back("--plugin");
    args.push_back(_default_ado_plugin);

    //    if (option_DEBUG > 2)
    PLOG("Launching ADO path: (%s), plugin (%s)", _default_ado_path.c_str(), _default_ado_plugin.c_str());

    ado = _i_ado_mgr->create(pool_id, _default_ado_path, args, 0);

    if (option_DEBUG > 2)
      PLOG("ADO process launched OK.");

    assert(ado);
    _ado_map[pool_id] = std::make_pair(ado, handler);
  }
  else {
    ado = i->second.first;
    bootstrap = false;
  }

  /* conditionally bootstrap ADO */
  if(bootstrap) {
    auto rc = ado->bootstrap_ado();
    if(rc != S_OK) {
      return rc;
    }

    /* exchange memory mapping information */
    if(nupm::check_mcas_kernel_module()) {

      std::vector<::iovec> regions;
      auto rc = _i_kvstore->get_pool_regions(pool_id, regions);
      if(rc != S_OK)
        return rc;

      for(auto& r: regions) {

        /* expose memory - for the moment use the address as the token */
        uint64_t token = reinterpret_cast<uint64_t>(r.iov_base);

        nupm::revoke_memory(token); /*remove any existing registrations */
        if(nupm::expose_memory(token,
                               r.iov_base,
                               r.iov_len) != S_OK) {
          PWRN("Shard: failed to expose memory to ADO");
          continue;
        }
        ado->send_memory_map(token, r.iov_len, r.iov_base);

        if (option_DEBUG > 2)
          PLOG("Shard: exposed region: %p %lu", r.iov_base, r.iov_len);
      }
    }


#if defined(PROFILE) && defined(PROFILE_POST_ADO)
    PLOG("Starting profiler");
    ProfilerStart("post_ado_launch.prof");
#endif
  } /* end of bootstrap */

  return S_OK;
}

void Shard::process_put_ado_request(Connection_handler* handler,
                                    Protocol::Message_put_ado_request* msg)
{
  Component::IADO_proxy * ado;
  status_t rc;
  
  static const auto error_func = [&](const char * message) {
    auto response_iob = handler->allocate();
    auto response = new (response_iob->base())
    Protocol::Message_ado_response(response_iob->length(),
                                   handler->auth_id(),
                                   msg->request_id,
                                   const_cast<char*>(message),
                                   strlen(message));
    response->set_status(E_INVAL);
    response_iob->set_length(response->message_size());
    handler->post_send_buffer(response_iob);
  };

#ifdef SHORT_CIRCUIT_ADO_HANDLING
  error_func("ADO!SC");
  return;
#endif

  if(!_i_ado_mgr) {
    error_func("ADO!NOT_ENABLED");
    return;
  }

  conditional_bootstrap_ado_process(handler, msg->pool_id, ado);

  assert(msg->value_len() > 0);

  rc = _i_kvstore->put(msg->pool_id,
                       msg->key(),
                       msg->value(),
                       msg->value_len());
  if(rc != S_OK)
    throw General_exception("put_ado_invoke: put failed");
  
  Component::IKVStore::key_t key_handle;
  auto locktype = Component::IKVStore::STORE_LOCK_ADO;
  void * value = nullptr;
  size_t value_len;
  
  if(_i_kvstore->lock(msg->pool_id,
                      msg->key(),
                      locktype,
                      value,
                      value_len,
                      key_handle) != S_OK) {
      error_func("ADO!ALREADY_LOCKED");
      return;
  }

  if(key_handle == Component::IKVStore::KEY_NONE)
    throw Logic_exception("lock gave KEY_NONE");

  if (option_DEBUG > 2)
    PLOG("Shard_ado: locked KV pair (value=%p, value_len=%lu)", value, value_len);

  /* register outstanding work */
  auto wr = _wr_allocator.allocate();
  *wr = { msg->pool_id, key_handle, locktype, msg->request_id, msg->flags };

  auto wr_key = reinterpret_cast<work_request_key_t>(wr); /* pointer to uint64_t */
  _outstanding_work.insert(wr_key);

  wmb();

  /* now send the work request */
  ado->send_work_request(wr_key,
                         msg->key(),
                         value, value_len,
                         msg->request(),
                         msg->request_len());

  if (option_DEBUG > 2)
    PLOG("Shard_ado: sent work request (len=%lu, key=%lx)", msg->request_len(), wr_key);
}

void Shard::process_ado_request(Connection_handler* handler,
                                Protocol::Message_ado_request* msg)
{
  Component::IADO_proxy * ado;
  
  static const auto error_func = [&](const char * message) {
    auto response_iob = handler->allocate();
    auto response = new (response_iob->base())
    Protocol::Message_ado_response(response_iob->length(),
                                   handler->auth_id(),
                                   msg->request_id,
                                   const_cast<char*>(message),
                                   strlen(message));
    response->set_status(E_INVAL);
    response_iob->set_length(response->message_size());
    handler->post_send_buffer(response_iob);
  };

#ifdef SHORT_CIRCUIT_ADO_HANDLING
  error_func("ADO!SC");
  return;
#endif

  if(!_i_ado_mgr) {
    error_func("ADO!NOT_ENABLED");
    return;
  }

  conditional_bootstrap_ado_process(handler, msg->pool_id, ado);

  /* get key-value pair */
  void * value = nullptr;
  size_t value_len = msg->ondemand_val_len;

  // if(value_len == 0) {
  //   error_func("ADO!E_INVAL");
  //   return;
  // }

  Component::IKVStore::key_t key_handle;
  auto locktype = Component::IKVStore::STORE_LOCK_ADO;
  if(_i_kvstore->lock(msg->pool_id,
                      msg->key(),
                      locktype,
                      value,
                      value_len,
                      key_handle) != S_OK)
    {
      error_func("ADO!ALREADY_LOCKED");
      return;
    }

  if(key_handle == Component::IKVStore::KEY_NONE)
    throw Logic_exception("lock gave KEY_NONE");

  if (option_DEBUG > 2)
    PLOG("Shard_ado: locked KV pair (value=%p, value_len=%lu)", value, value_len);

  /* register outstanding work */
  auto wr = _wr_allocator.allocate();
  *wr = { msg->pool_id, key_handle, locktype, msg->request_id, msg->flags };

  auto wr_key = reinterpret_cast<work_request_key_t>(wr); /* pointer to uint64_t */
  _outstanding_work.insert(wr_key);

  wmb();

  /* now send the work request */
  ado->send_work_request(wr_key,
                         msg->key(),
                         value, value_len,
                         msg->request(),
                         msg->request_len());

  if (option_DEBUG > 2)
    PLOG("Shard_ado: sent work request (len=%lu, key=%lx)", msg->request_len(), wr_key);

  /* for "asynchronous" calls we don't send a message
     for "synchronous call" we don't send a response to the client
     until the work completion has been picked up.  Of course this
     gives synchronous semantics on the client side.  We may need to
     extend this to asynchronous semantics for longer ADO
     operations */
}


/** 
 * Handle messages coming back from the ADO process
 * 
 */
void Shard::process_messages_from_ado()
{
  for(auto record: _ado_map) { /* for each ADO process */

    Component::IADO_proxy* ado = record.second.first;
    Connection_handler * handler = record.second.second;

    assert(ado);
    assert(handler);

    {
      void * response = nullptr;
      size_t response_len = 0;
      work_request_key_t request_key = 0;
      status_t response_status;

      /* work completion */
      while(ado->check_work_completions(request_key, response_status, response, response_len)) {

        auto work_item = _outstanding_work.find(request_key);
        if( work_item == _outstanding_work.end() )
          throw General_exception("bad record key from ADO (0x%" PRIx64 ")", request_key);

        auto request_record = request_key_to_record(request_key);

        if (option_DEBUG > 2)
          PMAJOR("Shard: collected WORK completion (request_record=%p)", request_record);

        _outstanding_work.erase(work_item);

        /* unlock the KV pair */
        if( _i_kvstore->unlock(request_record->pool,
                               request_record->key_handle) != S_OK)
          throw Logic_exception("unlock for KV after ADO work completion failed");

        if (option_DEBUG > 2)
          PLOG("Unlocked KV pair (pool=%lx, key_handle=%p)", request_record->pool, request_record->key_handle);

        /* unlock deferred locks, e.g., resulting from table operation create */
        {
          std::vector<Component::IKVStore::key_t> keys_to_unlock;
          ado->get_deferred_unlocks(request_key, keys_to_unlock);
          for(auto k: keys_to_unlock) {
            if(_i_kvstore->unlock(request_record->pool, k) != S_OK)
              throw Logic_exception("deferred unlock failed");
            PLOG("Shard: deferred unlock (%p)", static_cast<void*>(k));
          }
        }

        /* for async, save failed requests */
        if(request_record->is_async())  {

          /* if the ADO operation response is bad, safe it for
             later, otherwise don't do anything */
          if(response_status != S_OK) {
            if(option_DEBUG > 2)
              PWRN("Shard: saving ADO completion failure");
            _failed_async_requests.push_back(request_record);
          }
          else {
            if(option_DEBUG > 2)
              PWRN("Shard: async ADO completion OK!");
          }
        }
        else /* for sync, give response */
          {
            auto iob = handler->allocate();

            auto response_msg = new (iob->base())
              Protocol::Message_ado_response(iob->length(),
                                             handler->auth_id(),
                                             request_record->request_id,
                                             response, /* this will copy the response data */
                                             response_len);

            response_msg->set_status(S_OK);
            iob->set_length(response_msg->message_size());
            handler->post_send_buffer(iob);

            ::free(response); /* free response data allocated by ADO plugin */
          }
        _wr_allocator.free_wr(request_record);
      }
    }

    /* table requests (e.g., create new KV pair) */
    uint64_t work_id;
    int op;
    std::string key;
    size_t value_len;
    size_t align;
    void * addr;

    while(ado->check_table_ops(work_id, op, key, value_len, align, addr))
      {
        switch(op) {
        case Component::IADO_proxy::OP_CREATE:
        case Component::IADO_proxy::OP_OPEN:
          {
            PLOG("Shard: received table op create/open");
            Component::IKVStore::key_t key_handle;
            void * value = nullptr;

            if(_i_kvstore->lock(ado->pool_id(),
                                key,
                                Component::IKVStore::STORE_LOCK_ADO,
                                value,
                                value_len,
                                key_handle) != S_OK)
              throw General_exception("ADO OP_CREATE/OP_OPEN request failed");

            if(option_DEBUG > 2)
              PLOG("Shard: locked KV pair (%p, %p,%lu)", static_cast<void*>(key_handle), value, value_len);

            ado->add_deferred_unlock(work_id, key_handle);
            ado->send_table_op_response(S_OK, static_cast<void*>(value), value_len);
            break;
          }
        case Component::IADO_proxy::OP_ERASE:
          {
            if(option_DEBUG > 2)
              PLOG("Shard: received table op erase");
            status_t rc = _i_kvstore->erase(ado->pool_id(), key);
            if(rc != S_OK)
              PWRN("Shard_ado: OP_ERASE failed (key=%s)", key.c_str());
          
            ado->send_table_op_response(rc);
            break;
          }
        case Component::IADO_proxy::OP_RESIZE:
          {
            if(option_DEBUG > 2)
              PLOG("Shard: received table op resize value (work_id=%p)", work_id);
            
            status_t rc;
            work_request_t * wr = nullptr;
            /* check resize isn't on current work request. if so,
               we need to unlock and then relock */
            auto work_item = _outstanding_work.find(work_id);
          
            if(work_item != _outstanding_work.end()) {
              wr = request_key_to_record(work_id);
              assert(wr);
              if((rc = _i_kvstore->unlock(ado->pool_id(), wr->key_handle))!= S_OK) {
                ado->send_table_op_response(rc);
                break;
              }
              if(option_DEBUG > 2)
                PLOG("Shard: table op resize, unlocked");
            }
            else {
              ado->send_table_op_response(E_FAIL);
              break;
            }

            /* perform resize */
            void * new_value = nullptr;
            size_t new_value_len = 0;
            rc = _i_kvstore->resize_value(ado->pool_id(),
                                          key,
                                          value_len,
                                          align);

            /* relock if needded */
            if(wr)  {
              if(_i_kvstore->lock(ado->pool_id(),
                                  key,
                                  Component::IKVStore::STORE_LOCK_ADO,
                                  new_value,
                                  new_value_len,
                                  wr->key_handle) != S_OK)
                throw Logic_exception("ADO OP_RESIZE request failed to relock");

              if(option_DEBUG > 2)
                PLOG("Shard: table op resize, re-locked");
            }
          
            ado->send_table_op_response(rc, new_value, new_value_len);
            break;
          }
        case Component::IADO_proxy::OP_ALLOC:
          {
            status_t rc;
            // auto work_item = _outstanding_work.find(work_id);
            // if(work_item == _outstanding_work.end()) {
            //   ado->send_table_op_response(E_INVAL);
            //   break;
            // }

            void * out_addr = nullptr;
            rc = _i_kvstore->allocate_pool_memory(ado->pool_id(),
                                                 value_len,
                                                 align,
                                                 out_addr);

            if(option_DEBUG > 2)
              PLOG("Shard: allocate_pool_memory align=%lu rc=%d addr=%p",
                   align, rc, out_addr);
            ado->send_table_op_response(rc, out_addr);
            break;
          }
        case Component::IADO_proxy::OP_FREE:
          {
            status_t rc;
            auto work_item = _outstanding_work.find(work_id);
            if(work_item == _outstanding_work.end() || addr == nullptr) {
              ado->send_table_op_response(E_INVAL);
              break;
            }

            rc = _i_kvstore->free_pool_memory(ado->pool_id(), addr);
            if(option_DEBUG > 2)
              PLOG("Shard: allocate_pool_memory free rc=%d", rc);
            ado->send_table_op_response(rc);
            
            break;
          }
        default:
          throw Logic_exception("unknown table op code");
        }

      }
  }
}
