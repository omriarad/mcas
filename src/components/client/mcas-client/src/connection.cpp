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

#include "connection.h"

#include "protocol.h"
#include "range.h"

#include <city.h>
#include <common/cycles.h>
#include <common/delete_copy.h>
#include <common/utils.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric> /* accumulate */


#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/prettywriter.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <rapidjson/schema.h>
#pragma GCC diagnostic pop
#include <rapidjson/stringbuffer.h>
#include <rapidjson/error/error.h>  // rapidjson::ParseErrorCode

static constexpr const unsigned TLS_DEBUG_LEVEL = 3;

/* environment variables */
static constexpr const char* ENVIRONMENT_VARIABLE_CERT = "CERT";
static constexpr const char* ENVIRONMENT_VARIABLE_KEY = "KEY";
static constexpr const char* ENVIRONMENT_VARIABLE_SC = "SHORT_CIRCUIT_BACKEND";
static constexpr const char* ENVIRONMENT_VARIABLE_FORCE_DIRECT = "FORCE_DIRECT";

/* static constructor called once */
static void print_logs(int level, const char* msg) { printf("GnuTLS [%d]: %s", level, msg); }

static void __attribute__((constructor)) Global_ctor()
{
  gnutls_global_init();
  gnutls_global_set_log_level(0); /* 0-9 higher more verbose */
  gnutls_global_set_log_function(print_logs);
}

static constexpr const char * cipher_suite = "NORMAL:+AEAD";

namespace mcas
{
[[noreturn]] void throw_parse_exception(rapidjson::ParseErrorCode code, const char *msg, size_t offset);
}

//#define DEBUG_NPC_RESPONSES

using namespace component;

using memory_registered_fabric = mcas::memory_registered<IFabric_client>;

struct remote_fail : public std::runtime_error {
private:
  int _status;

public:
  remote_fail(int status_) : runtime_error("remote fail"), _status(status_) {}
  int status() const { return _status; }
};

namespace mcas
{
namespace client
{
/* The various embedded returns and throws suggest that the allocated
 * iobs should be automatically freed to avoid leaks.
 */

struct iob_free {
private:
  Connection_handler *_h;

public:
  iob_free(Connection_handler *h_) : _h(h_) {}
  void operator()(Connection_handler::buffer_t *iob) { _h->free_buffer(iob); }
};

struct memory_registered {
private:
	common::moveable_ptr<void> _desc;
	common::moveable_ptr<Registrar_memory_direct> _rmd;
	component::IMCAS::memory_handle_t _h;

public:
	memory_registered(Registrar_memory_direct * rmd_,
                          const mcas::range<char *> &range_  // range to register
                          , void *desc_
                          )
		: _desc(desc_)
		, _rmd( desc_ ? nullptr : rmd_)
		, _h(_rmd ? _rmd->register_direct_memory(range_.first, range_.length()) : IMCAS::MEMORY_HANDLE_NONE)
	{
		TM_SCOPE()
	}

	memory_registered(Registrar_memory_direct * rmd_,
                          const mcas::range<char *> &range_  // range to register
                          , component::IKVStore::memory_handle_t handle_
                          )
		: _desc( handle_ == IMCAS::MEMORY_HANDLE_NONE ? nullptr : static_cast<client::Fabric_transport::buffer_base *>(_h)->get_desc() )
		, _rmd( _desc ? nullptr : rmd_ )
		, _h(_rmd ? _rmd->register_direct_memory(range_.first, range_.length()) : IMCAS::MEMORY_HANDLE_NONE)
	{
	}
  DELETE_COPY(memory_registered);
  memory_registered(memory_registered &&) noexcept = default;
  virtual ~memory_registered()
  {
    if (_rmd) {
      _rmd->unregister_direct_memory(_h);
    }
  }
  void *desc() const { return _h == IMCAS::MEMORY_HANDLE_NONE ? _desc.get() : static_cast<client::Fabric_transport::buffer_base *>(_h)->get_desc(); }
};

struct mr_many
{
	std::vector<memory_registered> _vec;
	mr_many(Registrar_memory_direct *rmd_, gsl::span<const mcas::range<char *>> range_, gsl::span<const component::IKVStore::memory_handle_t> handles_)
		: _vec()
	{
		_vec.reserve(std::size(range_));
		for ( std::size_t i = 0; i != std::size(range_); ++i )
		{
			_vec.emplace_back(
				rmd_
				, range_[i]
				/* If no handle provided, or handle value is "HANDLE_NONE",
				 * Then ask memory_registered to create/destruct a one-time handle
				 * Else, use the provided handle.
				 */
				, handles_.size() <= i || handles_[i] == IKVStore::HANDLE_NONE ? nullptr : static_cast<client::Fabric_transport::buffer_base *>(handles_[i])->get_desc()
			);
		}
	}
	const memory_registered &at(std::size_t i) const { return _vec.at(i); }
};

/**
 * @brief Used to track buffers for asynchronous invocations
 *
 * is an async handle.
 */
struct async_buffer_set_t : public component::IMCAS::Opaque_async_handle, protected common::log_source {
  using iob_ptr = std::unique_ptr<client::Fabric_transport::buffer_t, iob_free>;

protected:
  iob_ptr        iobs;
  iob_ptr        iobr;

  async_buffer_set_t(unsigned debug_level_, iob_ptr &&iobs_, iob_ptr &&iobr_) noexcept
    : component::IMCAS::Opaque_async_handle{},
      common::log_source(debug_level_),
      iobs(std::move(iobs_)),
      iobr(std::move(iobr_))
  {
    CPLOG(2, "%s iobs %p iobr %p"
          , __func__
          , common::p_fmt(&*iobs)
          , common::p_fmt(&*iobr)
          );
  }

  async_buffer_set_t()                           = delete;
  DELETE_COPY(async_buffer_set_t);

public:
  virtual ~async_buffer_set_t() {}
  virtual int move_along(Connection_handler *c) = 0;
};

/* Nothing more than the two buffers. Used for async erase */
struct async_buffer_set_simple : public async_buffer_set_t {
  async_buffer_set_simple(unsigned debug_level_, iob_ptr &&iobs_, iob_ptr &&iobr_) noexcept
    : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_))
  {
  }
  int move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      iobs.reset(nullptr);
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }

      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC PUT");

      auto status = response_msg->get_status();
      iobr.reset(nullptr);
      return status;
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

struct async_buffer_set_get_locate
  : public async_buffer_set_t
  , public memory_registered {
private:
  static constexpr const char *_cname = "async_buffer_set_get_locate";
  iob_ptr                      _iobrd;
  component::IMCAS::pool_t     _pool;
  std::uint64_t                _auth_id;
  void *                       _value;
  std::size_t &                _value_len;
  void *                       _desc[1];
  ::iovec                      _v[1];
  std::uint64_t                _addr;

public:
  async_buffer_set_get_locate(TM_ACTUAL unsigned debug_level_,
                              Registrar_memory_direct *rmd_,
                              iob_ptr &&               iobrd_,
                              iob_ptr &&               iobs_,
                              iob_ptr &&               iobr_,
                              component::IMCAS::pool_t pool_,
                              std::uint64_t            auth_id_,
                              void *                   value_,
                              std::size_t &            value_len_, // In: buffer size of value_. Out: actual KV store value size
                              Connection_handler       *c,
                              void *                   desc_,
                              std::uint64_t            addr_,
                              std::uint64_t            key_
                              )
  : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_))
  , memory_registered(rmd_,
      mcas::range<char *>(static_cast<char *>(value_), static_cast<char *>(value_) + value_len_)
      .round_inclusive(4096),
      desc_),
    _iobrd(std::move(iobrd_)),
    _pool(pool_),
    _auth_id(auth_id_),
    _value(value_),
    _value_len(value_len_),
    _desc{this->desc()},  // provided by M
    _v{::iovec{_value, _value_len}},
    _addr(addr_)
  {
    TM_SCOPE()
    CPLOG(2, "%s: iobrd %p iobs2 %p iobr2 %p"
          , __func__
          , common::p_fmt(&*_iobrd)
          , common::p_fmt(&*iobs)
          , common::p_fmt(&*iobr)
          );
    /* reply have been received, with credentials for the DMA */

    CPLOG(2,
          "%s::%s post_read %p local (addr %p.%zx desc %p) <- (_addr 0x%zx, key 0x%zx)"
          , _cname
          , __func__
          , common::p_fmt(&*_iobrd)
          , _v[0].iov_base, _v[0].iov_len
          , _desc[0]
          , _addr, key_
          );
    c->post_read(std::begin(_v), std::end(_v), std::begin(_desc), _addr, key_, &*_iobrd);
    /* End */
  }

  DELETE_COPY(async_buffer_set_get_locate);

  int move_along(Connection_handler *c) override
  {
    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      CPLOG(2, "%s dma complete %p", __func__, common::p_fmt(&*_iobrd));
      _iobrd.reset(nullptr);
      /* DMA is complete. Issue GET_RELEASE */

      /* send release message */
      const auto msg = new (iobs->base())
        protocol::Message_IO_request(_auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_GET_RELEASE, _addr);

      c->post_recv(&*iobr);
      c->sync_inject_send(&*iobs, msg, __func__);
      /* End */
    }

    if ( iobr )
      {
        if ( ! c->test_completion(&*iobr) )
          {
            return E_BUSY;
          }
        /* What to do when second recv completes */
        const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC GET_RELEASE");
        auto status = response_msg->get_status();

        iobr.reset(nullptr);
        return status;
        /* End */
      }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

namespace
{
	/* Wanted: a generator which takes a range r and a function f and returns
	 * values of f(r). Probably in boost ...
	 */
	std::vector<mcas::range<char *>> make_rounded_range_vector(gsl::span<const common::const_byte_span> values_, std::size_t round_)
	{
		std::vector<mcas::range<char *>> v;
		v.reserve(values_.size());
		for ( std::size_t i = 0; i != values_.size(); ++i )
		{
			v.emplace_back(
				static_cast<char *>(const_cast<void *>(::base(values_[i])))
				, static_cast<char *>(const_cast<void *>(::end(values_[i])))
			);
			v.back().round_inclusive(round_);
		}
		return v;
	}
}

struct async_buffer_set_put_locate
	: public async_buffer_set_t
	, private mr_many
{
private:
  static constexpr const char *_cname = "async_buffer_set_put_locate";
  iob_ptr                      _iobrd;
  iob_ptr                      _iobs2;
  iob_ptr                      _iobr2;
  component::IMCAS::pool_t     _pool;
  std::uint64_t                _auth_id;
  std::vector<void *>          _desc;
  std::vector<::iovec>         _v;
  std::uint64_t                _addr;

public:
  async_buffer_set_put_locate(TM_ACTUAL unsigned                 debug_level_,
                              Registrar_memory_direct *rmd_,
                              iob_ptr &&               iobs_,
                              iob_ptr &&               iobr_,
                              iob_ptr &&               iobrd_,
                              iob_ptr &&               iobs2_,
                              iob_ptr &&               iobr2_,
                              component::IMCAS::pool_t pool_,
                              std::uint64_t            auth_id_,
                              gsl::span<const common::const_byte_span> values_,
      gsl::span<const component::IKVStore::memory_handle_t> handles_
    )
  : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
    mr_many(rmd_,
      make_rounded_range_vector(values_, 4096),
      handles_
    ),
    _iobrd(std::move(iobrd_)),
    _iobs2(std::move(iobs2_)),
    _iobr2(std::move(iobr2_)),
    _pool{pool_},
    _auth_id{auth_id_},
	_desc()
	, _v()
    , _addr{}
  {
    _v.reserve(values_.size());
    _desc.reserve(values_.size());
    CPLOG(2, "%s: iobrd %p iobs2 %p iobr2 %p"
          , __func__
          , common::p_fmt(&*_iobrd)
          , common::p_fmt(&*_iobs2)
          , common::p_fmt(&*_iobr2)
          );
    for ( std::size_t i = 0; i != values_.size(); ++i )
    {
      /* There is no intention to modify the source, but libfabric uses vectors of
	   * iovec, whic does not have a "const ptr" version
	   */
      _v.emplace_back(::iovec{const_cast<void *>(::base(values_[i])), ::size(values_[i])});
      _desc.emplace_back(this->at(i).desc());
    }
  }
  DELETE_COPY(async_buffer_set_put_locate);
  int                  move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      /* What to do when first send completes */
      iobs.reset(nullptr);
      /* End */
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      /* What to do when first recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC PUT_LOCATE");
      auto status = response_msg->get_status();

      _addr    = response_msg->addr;
      auto key = response_msg->key;
      iobr.reset(nullptr);
      if (status != S_OK) {
        return status;
      }
      /* reply have been received, with credentials for the DMA */

      CPLOG(2,
            "%s post_write %p -> (_addr 0x%zx, key 0x%zx)"
            , __func__
            , common::p_fmt(&*_iobrd)
            , _addr, key
            );
      for ( std::size_t i = 0; i != _v.size(); ++i )
      {
         CPLOG(2,
            "%s post_write local (addr %p.%zx desc %p)"
            , __func__
            , _v[i].iov_base, _v[i].iov_len
            , _desc[i]
            );
      }

      c->post_write(_v, &*_desc.begin(), _addr, key, &*_iobrd);
      /* End */
    }

    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      CPLOG(2, "%s dma complete %p", __func__, common::p_fmt(&*_iobrd));
      _iobrd.reset(nullptr);
      /* DMA is complete. Issue PUT_RELEASE */

      /* send release message */
      const auto msg = new (_iobs2->base())
        protocol::Message_IO_request(_auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_PUT_RELEASE, _addr);

      c->post_recv(&*_iobr2);
      c->sync_inject_send(&*_iobs2, msg, __func__);
      /* End */
    }

    if (_iobr2) {
      if (!c->test_completion(&*_iobr2)) {
        return E_BUSY;
      }
      /* What to do when second recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*_iobr2, "ASYNC PUT_RELEASE");
      auto status = response_msg->get_status();

      _iobr2.reset(nullptr);
      return status;
      /* End */
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

struct async_buffer_set_invoke : public async_buffer_set_t {
  std::vector<IMCAS::ADO_response> *out_ado_response;

public:
  async_buffer_set_invoke(unsigned                          debug_level_,
                          iob_ptr &&                        iobs_,
                          iob_ptr &&                        iobr_,
                          std::vector<IMCAS::ADO_response> *out_ado_response_)
    : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
      out_ado_response(out_ado_response_)
  {
  }
  DELETE_COPY(async_buffer_set_invoke);
  int              move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      iobs.reset(nullptr);
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      return c->receive_and_process_ado_response(iobr, *out_ado_response);
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

struct async_buffer_set_get_direct_offset
  : public async_buffer_set_t
  , public memory_registered {
private:
  using locate_element                               = protocol::Message_IO_response::locate_element;
  static constexpr const char *               _cname = "async_buffer_set_get_direct_offset";
  iob_ptr                                     _iobrd;
  iob_ptr                                     _iobs2;
  iob_ptr                                     _iobr2;
  component::IMCAS::pool_t                    _pool;
  std::uint64_t                               _auth_id;
  std::size_t                                 _offset;
  char *                                      _buffer;
  std::size_t &                               _length;
  std::uint64_t                               _key;
  void *                                      _desc[1];
  ::iovec                                     _v[1];
  std::vector<locate_element>                 _addr_list;
  std::vector<locate_element>::const_iterator _addr_cursor;

public:
  async_buffer_set_get_direct_offset(TM_ACTUAL unsigned                 debug_level_,
                                     Registrar_memory_direct *rmd_,
                                     iob_ptr &&               iobs_,
                                     iob_ptr &&               iobr_,
                                     iob_ptr &&               iobrd_,
                                     iob_ptr &&               iobs2_,
                                     iob_ptr &&               iobr2_,
                                     component::IMCAS::pool_t pool_,
                                     std::uint64_t            auth_id_,
                                     std::size_t              offset_,
                                     void *                   buffer_,
                                     std::size_t &            length_,
                                     void *                   desc_)
  : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
    memory_registered(rmd_,
      mcas::range<char *>(static_cast<char *>(buffer_), static_cast<char *>(buffer_) + length_)
      .round_inclusive(4096),
      desc_),
    _iobrd(std::move(iobrd_)),
    _iobs2(std::move(iobs2_)),
    _iobr2(std::move(iobr2_)),
    _pool(pool_),
    _auth_id{auth_id_},
    _offset{offset_},
    _buffer(static_cast<char *>(buffer_)),
    _length(length_),
    _key{},
    _desc{this->desc()}  // provided by M
  ,
    _v{},
    _addr_list{},
    _addr_cursor{}
  {
    CPLOG(2, "%s iobs2 %p iobr2 %p"
          , __func__
          , common::p_fmt(&*_iobs2)
          , common::p_fmt(&*_iobr2)
          );
  }
  DELETE_COPY(async_buffer_set_get_direct_offset);
  int                         move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      /* What to do when first send completes */
      iobs.reset(nullptr);
      /* End */
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      /* What to do when first recv completes */
      const auto response = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC OP_LOCATE");
      auto status = response->get_status();
      {
        auto cursor = response->edata();
        _addr_list  = std::vector<locate_element>(cursor, cursor + response->element_count());

        CPLOG(2,
              "%s::%s: edata count %zu %p to %p"
              , _cname
              , __func__
              , response->element_count()
              , common::p_fmt(cursor)
              , common::p_fmt(cursor + response->element_count())
              );

        _length = 0;
        for (const auto &e : _addr_list) {
          _length += e.len;
          CPLOG(2, "%s::%s: addr 0x%" PRIx64 " len 0x%" PRIx64
                , _cname
                , __func__
                , e.addr, e.len);
        }
      }
      _key         = response->key;
      _addr_cursor = _addr_list.begin();

      iobr.reset(nullptr);

      return status == S_OK ? E_BUSY : status;
    }

    if ( _addr_list.empty() )
      {
        return S_OK;
      }

    if (_addr_cursor != _addr_list.end()) {
      if (_iobrd && !c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }

      _iobrd = c->make_iob_ptr_read();
      CPLOG(2, "%s iobrd %p"
            , __func__
            , common::p_fmt(&*_iobrd)
            );

      /* reply have been received, with credentials for the DMA */
      _v[0] = ::iovec{_buffer, _addr_cursor->len};

      CPLOG(2,
            "%s::%s post_read %p local (addr %p.%zx desc %p) <- (_addr 0x%zx, key 0x%zx)"
            , _cname
            , __func__
            , common::p_fmt(&*_iobrd)
            , _v[0].iov_base, _v[0].iov_len
            , _desc[0]
            , _addr_cursor->addr, _key
            );
      c->post_read(std::begin(_v), std::end(_v), std::begin(_desc), _addr_cursor->addr, _key, &*_iobrd);
      _buffer += _addr_cursor->len;
      ++_addr_cursor;
      /* End */
    }

    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      /* DMA done. Might need another DMA */
      CPLOG(2, "%s::%s dma read complete %p"
            , _cname
            , __func__
            , common::p_fmt(&*_iobrd)
            );

      _iobrd.reset(nullptr);
      /* DMA is complete. Issue OP_RELEASE */

      /* send release message */
      const auto msg = new (_iobs2->base()) protocol::Message_IO_request(
                                                                         _auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_RELEASE, _offset, _length);

      c->post_recv(&*_iobr2);
      c->sync_inject_send(&*_iobs2, msg, __func__);
      /* End */
    }

    /* release in process, or not needed because length is 0 */
    if ( _iobr2 ) {
      if ( _iobr2 && ! c->test_completion(&*_iobr2) ) {
        return E_BUSY;
      }
      /* What to do when second recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*_iobr2, "OP_RELEASE");
      return response_msg->get_status();
      /* End */
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

struct async_buffer_set_put_direct_offset
  : public async_buffer_set_t
  , public memory_registered {
private:
  using locate_element                               = protocol::Message_IO_response::locate_element;
  static constexpr const char *               _cname = "async_buffer_set_put_direct_offset";
  iob_ptr                                     _iobrd;
  iob_ptr                                     _iobs2;
  iob_ptr                                     _iobr2;
  component::IMCAS::pool_t                    _pool;
  std::uint64_t                               _auth_id;
  std::size_t                                 _offset;
  const char *                                _buffer;
  std::size_t &                               _length;
  std::uint64_t                               _key;
  void *                                      _desc[1];
  ::iovec                                     _v[1];
  std::vector<locate_element>                 _addr_list;
  std::vector<locate_element>::const_iterator _addr_cursor;

public:
  async_buffer_set_put_direct_offset(TM_ACTUAL unsigned                 debug_level_,
                                     Registrar_memory_direct *rmd_,
                                     iob_ptr &&               iobs_,
                                     iob_ptr &&               iobr_,
                                     iob_ptr &&               iobrd_,
                                     iob_ptr &&               iobs2_,
                                     iob_ptr &&               iobr2_,
                                     component::IMCAS::pool_t pool_,
                                     std::uint64_t            auth_id_,
                                     std::size_t              offset_,
                                     const void *             buffer_,
                                     std::size_t &            length_,
                                     void *                   desc_)
  : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
    memory_registered(rmd_,
      mcas::range<char *>(static_cast<char *>(const_cast<void *>(buffer_)),
                          static_cast<char *>(const_cast<void *>(buffer_)) + length_)
      .round_inclusive(4096),
      desc_),
    _iobrd(std::move(iobrd_)),
    _iobs2(std::move(iobs2_)),
    _iobr2(std::move(iobr2_)),
    _pool(pool_),
    _auth_id{auth_id_},
    _offset{offset_},
    _buffer(static_cast<const char *>(buffer_)),
    _length(length_),
    _key{},
    _desc{this->desc()}  // provided by M
  ,
    _v{},
    _addr_list{},
    _addr_cursor{}
  {
    CPLOG(2, "%s iobs2 %p iobr2 %p"
          , __func__
          , common::p_fmt(&*_iobs2)
          , common::p_fmt(&*_iobr2)
          );
  }
  DELETE_COPY(async_buffer_set_put_direct_offset);
  int                         move_along(Connection_handler *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(&*iobs) == false) {
        return E_BUSY;
      }
      /* What to do when first send completes */
      iobs.reset(nullptr);
      /* End */
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(&*iobr) == false) {
        return E_BUSY;
      }
      /* What to do when first recv completes */
      const auto response = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC OP_LOCATE");
      auto status = response->get_status();

      {
        auto cursor = response->edata();
        _addr_list  = std::vector<locate_element>(cursor, cursor + response->element_count());
        CPLOG(2,
              "%s::%s: edata count %zu %p to %p"
              , _cname
              , __func__
              , response->element_count()
              , common::p_fmt(cursor)
              , common::p_fmt(cursor + response->element_count())
              );
        _length = 0;
        for (const auto &e : _addr_list) {
          _length += e.len;
          CPLOG(2,
                "%s::%s: addr 0x%" PRIx64 " len 0x%" PRIx64
                , _cname
                , __func__
                , e.addr, e.len
                );
        }
      }

      _key         = response->key;
      _addr_cursor = _addr_list.begin();

      iobr.reset(nullptr);

      return status == S_OK ? E_BUSY : status;
    }

    if ( _addr_list.empty() )
      {
        return S_OK;
      }

    if (_addr_cursor != _addr_list.end()) {
      if (_iobrd && !c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }

      _iobrd = c->make_iob_ptr_write();
      CPLOG(2, "%s iobrd %p"
            , __func__
            , common::p_fmt(&*_iobrd)
            );

      /* reply received, with credentials for the DMA */
      _v[0] = ::iovec{const_cast<char *>(_buffer), _addr_cursor->len};

      CPLOG(2,
            "%s::%s post_write %p local (addr %p.%zx desc %p) -> (_addr 0x%zx, key 0x%zx)"
            , _cname
            , __func__
            , common::p_fmt(&*_iobrd)
            , _v[0].iov_base, _v[0].iov_len
            , _desc[0]
            , _addr_cursor->addr
            , _key
            );

      c->post_write(_v, std::begin(_desc), _addr_cursor->addr, _key, &*_iobrd);
      _buffer += _addr_cursor->len;
      ++_addr_cursor;
      /* End */
    }

    if (_iobrd) {
      if (!c->test_completion(&*_iobrd)) {
        return E_BUSY;
      }
      /* What to do when DMA completes */
      /* DMA done. Might need another DMA */
      CPLOG(2
            , "%s::%s dma write complete %p"
            , _cname
            , __func__
            , common::p_fmt(&*_iobrd)
            );

      _iobrd.reset(nullptr);
      /* DMA is complete. Issue OP_RELEASE */

      /* send release message */
      const auto msg = new (_iobs2->base()) protocol::Message_IO_request(
                                                                         _auth_id, c->request_id(), _pool, protocol::OP_TYPE::OP_RELEASE_WITH_FLUSH, _offset, _length);

      c->post_recv(&*_iobr2);
      c->sync_inject_send(&*_iobs2, msg, __func__);
      /* End */
    }

    if ( _iobr2 ) {
      if ( ! c->test_completion(&*_iobr2) ) {
        return E_BUSY;
      }
      /* What to do when second recv completes */
      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*_iobr2, "OP_RELEASE_WITH_FLUSH,");
      return response_msg->get_status();
      /* End */
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

Connection_handler::Connection_handler(const unsigned              debug_level,
                                       Connection_base::Transport *connection,
                                       Connection_base::buffer_manager &bm_,
                                       const unsigned              patience,
                                       const common::string_view   other)
  : Connection_base(debug_level, connection, bm_, patience),
#ifdef THREAD_SAFE_CLIENT
    _api_lock{},
#endif
    _exit{false},
    _request_id{0},
    _max_message_size{0},
    _max_inject_size(connection->max_inject_size()),
    _options()
{
  char *env = ::getenv(ENVIRONMENT_VARIABLE_SC);
  if (env && env[0] == '1') {
    _options.short_circuit_backend = true;
  }

  if(::getenv(ENVIRONMENT_VARIABLE_KEY) && ::getenv(ENVIRONMENT_VARIABLE_CERT))
    _options.tls = true;

  if(::getenv(ENVIRONMENT_VARIABLE_FORCE_DIRECT))
    _force_direct = true;

  if(other.data()) {
    try {
      rapidjson::Document doc;
      doc.Parse(other.data());
      if ( doc.HasParseError() )
      {
        throw std::domain_error{std::string{"JSON parse error \""} + rapidjson::GetParseError_En(doc.GetParseError()) + "\" at " + std::to_string(doc.GetErrorOffset())};
      }
      auto security = doc.FindMember("security");

      /* set security options */
      if(security != doc.MemberEnd() && doc["security"].IsString()) {
        std::string option(doc["security"].GetString());

        //PNOTICE("!!!!! option=%s", option.c_str());
        /* tls:auth indicates TLS authentication */
        if(option == "tls:auth") {
          _options.tls = true;
        }
      }
    }
    catch (...) {
      throw API_exception("extra configuration string parse failed");
    }
  }

}

Connection_handler::~Connection_handler()
{
  PLOG("%s: (%p)", __func__, common::p_fmt(this));
}

void Connection_handler::send_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, common::p_fmt(iob));
}
void Connection_handler::recv_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, common::p_fmt(iob));
}
void Connection_handler::write_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, common::p_fmt(iob));
}
void Connection_handler::read_complete(void *param, buffer_t *iob)
{
  PLOG("%s param %p iob %p", __func__, param, common::p_fmt(iob));
}

Connection_handler::pool_t Connection_handler::open_pool(const std::string name,
                                                         const unsigned int flags,
                                                         const addr_t base)
{
  API_LOCK();

  PMAJOR("Open pool: %s (flags=%u, base=0x%lx)", name.c_str(), flags, base);

  /* send pool request message */

  /* Use unique_ptr to ensure that the dynamically buffers are freed.
   * unique_ptr protects against the code forgetting to call free_buffer,
   * which it usually did when the function exited by a throw or a
   * non-terminal return.
   *
   */
  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  IKVStore::pool_t pool_id;

  assert(&*iobr != &*iobs);

  try {
    const auto msg =
      new (iobs->base())
      mcas::protocol::Message_pool_request(iobs->length(), auth_id(), /* auth id */
                                           request_id(), 0,           /* size */
                                           0,                         /* expected obj cnt */
                                           mcas::protocol::OP_OPEN,
                                           name,
                                           0, /* flags */
                                           base /* base virtual address */
                                           );

    /* The &* notation extracts a raw pointer form the "unique_ptr".
     * The difference is that the standard pointer does not imply
     * ownership; the function is simply "borrowing" the pointer.
     * Here, &*iobr is equivalent to iobr->get(). The choice is a
     * matter of style. &* uses two fewer tokens.
     */
    /* the sequence "post_recv, sync_inject_send, wait_for_completion"
     * is common enough that it probably deserves its own function.
     * But we may find that the entire single-exchange pattern as seen
     * in open_pool, create_pool and several others could be placed in
     * a single template function.
     */

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr); /* await response */

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    pool_id = response_msg->pool_id;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    pool_id = IKVStore::POOL_ERROR;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    pool_id = IKVStore::POOL_ERROR;
  }
  return pool_id;
}

Connection_handler::pool_t
Connection_handler::create_pool(const std::string  name,
                                const size_t       size,
                                const unsigned int flags,
                                const uint64_t     expected_obj_count,
                                const addr_t       base)
{
  API_LOCK();

  PMAJOR("Create pool: %s (flags=%u, base=0x%lx)", name.c_str(), flags, base);

  /* send pool request message */
  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  IKVStore::pool_t pool_id;

  try {
    const auto msg = new (iobs->base())
      protocol::Message_pool_request(iobs->length(),
                                     auth_id(),
                                     request_id(),
                                     size,
                                     expected_obj_count,
                                     mcas::protocol::OP_CREATE,
                                     name,
                                     flags,
                                     base);
    assert(msg->op());

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    pool_id = response_msg->pool_id;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    pool_id = IKVStore::POOL_ERROR;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    pool_id = IKVStore::POOL_ERROR;
  }

  /* Note: most request/response pairs return status. This one returns a pool_id
   * instead. */
  return pool_id;
}

status_t Connection_handler::close_pool(const pool_t pool)
{
  PMAJOR("Close pool: 0x%lx", pool);

  API_LOCK();
  /* send pool request message */
  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  const auto msg  = new (iobs->base())
    mcas::protocol::Message_pool_request(iobs->length(), auth_id(), request_id(), mcas::protocol::OP_CLOSE, pool);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    const auto status = response_msg->get_status();
    return status;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::delete_pool(const std::string &name)

{
  if (name.empty()) return E_INVAL;

  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  const auto msg = new (iobs->base())
    mcas::protocol::Message_pool_request(iobs->length(),
                                         auth_id(),
                                         request_id(),
                                         0, // size
                                         0, // exp obj count
                                         mcas::protocol::OP_DELETE,
                                         name,
                                         0, // flags
                                         0); // base

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    return response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::delete_pool(const IMCAS::pool_t pool)
{
  if (!pool) return E_INVAL;

  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  const auto msg = new (iobs->base())
    mcas::protocol::Message_pool_request(iobs->length(), auth_id(), request_id(), mcas::protocol::OP_DELETE, pool);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_pool_response>(&*iobr, __func__);

    return response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::configure_pool(const IMCAS::pool_t pool, const std::string &json)
{
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  if (!mcas::protocol::Message_IO_request::would_fit(json.length(), iobs->original_length())) {
    return IKVStore::E_TOO_LARGE;
  }

  const auto msg = new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                                         mcas::protocol::OP_CONFIGURE,  // op
                                                                         json);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  try {
    wait_for_completion(&*iobr);
    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

    return response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

/**
 * Memcpy version; both key and value are copied
 *
 */
status_t Connection_handler::put(const pool_t       pool,
                                 const std::string  key,
                                 const void *       value,
                                 const size_t       value_len,
                                 const unsigned int flags)
{
  if (value == nullptr || value_len == 0) return E_INVAL;
  return put(pool, key.c_str(), key.length(), value, value_len, flags);
}

status_t Connection_handler::put(const pool_t       pool,
                                 const void *       key,
                                 const size_t       key_len,
                                 const void *       value,
                                 const size_t       value_len,
                                 const unsigned int flags)
{
  TM_ROOT();
  API_LOCK();

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();

  if (debug_level() > 1)
    PINF("put: %.*s (key_len=%lu) (value_len=%lu)", int(key_len), static_cast<const char *>(key), key_len, value_len);

  /* check key length */
  if (!mcas::protocol::Message_IO_request::would_fit(key_len + value_len, iobs->original_length())) {
    PWRN("mcas_client::%s value length (%lu) too long. Use put_direct.", __func__, value_len);
    return IKVStore::E_TOO_LARGE;
  }

  status_t status;

  try {
    const auto msg =
      new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                            mcas::protocol::OP_PUT,  // op
                                                            key, key_len, value, value_len, flags);

    if (_options.short_circuit_backend) msg->add_scbe();

    iobs->set_length(msg->msg_len());

    post_recv(&*iobr);
    sync_send(&*iobs, msg, __func__); /* this will clean up iobs */
    {
      TM_SCOPE(wait_recv)
      wait_for_completion(&*iobr);
    }

    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

    status = response_msg->get_status();
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

auto Connection_handler::locate(const pool_t pool_, const std::size_t offset_, const std::size_t size_)
  -> std::tuple<uint64_t, std::vector<locate_element>>
{
  const auto iobr = make_iob_ptr_recv();
  const auto iobs = make_iob_ptr_send();

  /* send advance leader message */
  const auto msg = new (iobs->base())
    protocol::Message_IO_request(auth_id(), pool_, request_id(), protocol::OP_LOCATE, offset_, size_);

  post_recv(&*iobr);
  sync_inject_send(&*iobs, msg, __func__);
  /* wait for response from header before posting the value */
  wait_for_completion(&*iobr);

  const auto response = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

  if (response->get_status() != S_OK) {
    throw remote_fail(response->get_status());
  }

  auto                        cursor = response->edata();
  std::vector<locate_element> addr_list(cursor, cursor + response->element_count());

  return std::tuple<uint64_t, std::vector<locate_element>>(response->key, std::move(addr_list));
}

namespace
{
  std::size_t values_size(gsl::span<const common::const_byte_span> values_)
  {
    return std::accumulate(
      values_.begin(), values_.end()
      , 0
      , [] (std::size_t a, const common::const_byte_span &s) -> std::size_t
        {
          return a + size(s);
        }
    );
  }
}

IMCAS::async_handle_t Connection_handler::put_locate_async(TM_ACTUAL const pool_t                        pool,
                                                           const void *                        key,
                                                           const size_t                        key_len,
                                                           const gsl::span<const common::const_byte_span> values,
                                                           component::Registrar_memory_direct *rmd_,
                                                           gsl::span<const component::IKVStore::memory_handle_t> mem_handles_,
                                                           const unsigned                      flags)
{
  TM_SCOPE()
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* send locate message */
  const auto msg = new (iobs->base()) protocol::Message_IO_request(
                                                                   iobs->length(), auth_id(), request_id(), pool, protocol::OP_PUT_LOCATE, key, key_len, values_size(values), flags);
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  /*
   * The entire put_locate protocol involves five completions at the client:
   *   send PUT_LOCATE request
   *   recv PUT_LOCATE response
   *   write DMA
   *   send PUT_RELEASE request
   *   recv PUT_RELEASE response
   */
  return
    static_cast<IMCAS::async_handle_t>(
      new async_buffer_set_put_locate(
        TM_REF debug_level()
        , rmd_
        , std::move(iobs)
        , std::move(iobr)
        , make_iob_ptr_write()
        , make_iob_ptr_send()
        , make_iob_ptr_recv()
        , pool, auth_id()
        , values
        , mem_handles_
      )
    );
}

IMCAS::async_handle_t Connection_handler::get_direct_offset_async(const pool_t                        pool_,
                                                                  const std::size_t                   offset_,
                                                                  void *const                         buffer_,
                                                                  std::size_t &                       len_,
                                                                  component::Registrar_memory_direct *rmd_,
                                                                  void *const                         desc_)
{
  TM_ROOT()
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* send advance leader message */
  const auto msg = new (iobs->base())
    protocol::Message_IO_request(auth_id(), request_id(), pool_, protocol::OP_LOCATE, offset_, len_);
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  /*
   * The entire get_direct_offset protocol involves five completions at the
   * client: send LOCATE request recv LOCATE response read DMA send RELEASE
   * request recv RELEASE response
   */
  return
    static_cast<IMCAS::async_handle_t>(new async_buffer_set_get_direct_offset(
                                                                                                         TM_REF debug_level(), rmd_, std::move(iobs), std::move(iobr), iob_ptr(nullptr, this), make_iob_ptr_send(),
                                                                                                         make_iob_ptr_recv(), pool_, auth_id(), offset_, buffer_, len_, desc_));
}

IMCAS::async_handle_t Connection_handler::put_direct_offset_async(const pool_t                        pool_,
                                                                  const std::size_t                   offset_,
                                                                  const void *const                   buffer_,
                                                                  std::size_t &                       length_,
                                                                  component::Registrar_memory_direct *rmd_,
                                                                  void *const                         desc_)
{
  TM_ROOT()
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* send locate message */
  const auto msg = new (iobs->base())
    protocol::Message_IO_request(auth_id(), request_id(), pool_, protocol::OP_LOCATE, offset_, length_);
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  /*
   * The entire get_direct_offset protocol involves five completions at the
   * client: send LOCATE request recv LOCATE response read DMA send RELEASE
   * request recv RELEASE response
   */
  return static_cast<IMCAS::async_handle_t>
    (new async_buffer_set_put_direct_offset(TM_REF debug_level(),
                                                                         rmd_,
                                                                         std::move(iobs),
                                                                         std::move(iobr),
                                                                         iob_ptr(nullptr, this),
                                                                         make_iob_ptr_send(),
                                                                         make_iob_ptr_recv(),
                                                                         pool_,
                                                                         auth_id(),
                                                                         offset_,
                                                                         buffer_,
                                                                         length_,
                                                                         desc_));
}

IMCAS::async_handle_t
Connection_handler::get_locate_async(TM_ACTUAL const pool_t                        pool,
                                     const void *                        key,
                                     const size_t                        key_len,
                                     void *const                         value,
                                     size_t &                            value_len,
                                     component::Registrar_memory_direct *rmd_,
                                     void *const                         desc_,
                                     const unsigned                      flags)
{
  TM_SCOPE()
  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();
  const auto buffer_len = value_len;

  /* send advance leader message */
  const auto msg = new (iobs->base())
    protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool, protocol::OP_GET_LOCATE, key, key_len, value_len, flags);
  
  {
  TM_SCOPE(1)
  iobs->set_length(msg->msg_len());

  post_recv(&*iobr);
  post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

  {
  TM_SCOPE(wait_send)
  wait_for_completion(&*iobs);
  }
  iobs.reset(nullptr);

  {
  TM_SCOPE(wait_recv)
  wait_for_completion(&*iobr);
  }
  }
  const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC GET_LOCATE");
  auto status = response_msg->get_status();
  if (status != S_OK) {
    throw remote_fail(status);
  }

  value_len = response_msg->data_length();
  auto addr = response_msg->addr;
  auto memory_key = response_msg->key;
  iobs.reset(nullptr);
  auto transfer_len = std::min(buffer_len, value_len);

  {
  TM_SCOPE(2)
  /*
   * The entire get_locate protocol involves five completions at the client:
   *   send GET_LOCATE request
   *   recv GET_LOCATE response
   *   read DMA
   *   send GET_RELEASE request
   *   recv GET_RELEASE response
   */
  return static_cast<IMCAS::async_handle_t>(new async_buffer_set_get_locate(TM_REF debug_level(),
                                                                                                                 rmd_,
                                                                                                                 make_iob_ptr_read(),
                                                                                                                 make_iob_ptr_send(),
                                                                                                                 make_iob_ptr_recv(),
                                                                                                                 pool,
                                                                                                                 auth_id(),
                                                                                                                 value,
                                                                                                                 transfer_len,
                                                                                                                 this,
                                                                                                                 desc_,
                                                                                                                 addr,
                                                                                                                 memory_key));
  }
}

status_t Connection_handler::put_direct(pool_t                               pool_,
                      const void *                         key_,
                      size_t                               key_len_,
                      gsl::span<const common::const_byte_span> values_,
                      component::Registrar_memory_direct * rmd_,
                      gsl::span<const component::IMCAS::memory_handle_t> handles_,
                      const component::IMCAS::flags_t      flags_)
{
  TM_ROOT()
  component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;
  auto status = async_put_direct(pool_, key_, key_len_, values_, async_handle, rmd_, handles_, flags_);
  if (status == S_OK) {
    TM_SCOPE(spin)
    do {
      status = check_async_completion(async_handle);
    } while (status == E_BUSY);
  }
  return status;
}

status_t Connection_handler::async_put(const IMCAS::pool_t    pool,
                                       const void *           key,
                                       const size_t           key_len,
                                       const void *           value,
                                       const size_t           value_len,
                                       IMCAS::async_handle_t &out_handle,
                                       const IMCAS::flags_t   flags)
{
  API_LOCK();

  if (debug_level() > 1)
    PINF("%s: %.*s (key_len=%lu) (value_len=%lu)", __func__, int(key_len), static_cast<const char *>(key), key_len,
         value_len);

  auto iobr = make_iob_ptr_recv();
  auto iobs = make_iob_ptr_send();

  /* check key length */
  if (!mcas::protocol::Message_IO_request::would_fit(key_len + value_len, iobs->original_length())) {
    PWRN("mcas_client::%s value length (%lu) too long. Use async_put_direct.", __func__, value_len);
    return IKVStore::E_TOO_LARGE;
  }

  try {
    const auto msg =
      new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                            mcas::protocol::OP_PUT,  // op
                                                            key, key_len, value, value_len, flags);

    iobs->set_length(msg->msg_len());

    /* post both send and receive */
    post_recv(&*iobr);
    post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

    out_handle = new async_buffer_set_simple(debug_level(), std::move(iobs), std::move(iobr));

    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
  }

  return E_FAIL;
}

status_t Connection_handler::async_put_direct(const IMCAS::pool_t                        pool_,
                                              const void *const                          key_,
                                              const size_t                               key_len_,
                                              const gsl::span<const common::const_byte_span>   values_,
                                              component::IMCAS::async_handle_t &         out_async_handle_,
                                              component::Registrar_memory_direct *       rmd_,
                                              const gsl::span<const component::IKVStore::memory_handle_t> mem_handles_,
                                              const component::IMCAS::flags_t            flags_)
{
  TM_ROOT()
  API_LOCK();

  assert(_max_message_size);

  if (pool_ == 0) {
    PWRN("%s: invalid pool identifier", __func__);
    return E_INVAL;
  }

  for ( const auto & v : values_ )
  {
    if ( ::size(v) && ! ::base(v) ) {
      PWRN("%s: bad parameter value=%p value_len=%zu", __func__, ::base(v), ::size(v) );
      return E_BAD_PARAM;
    }
  }

  try {
    auto iobr = make_iob_ptr_recv();
    auto iobs = make_iob_ptr_send();

    if (values_.size() == 1 /* A simplification. We could change the small put code to handle multiple source */
        &&
        (_force_direct == false) &&
        (mcas::protocol::Message_IO_request::would_fit(key_len_ + ::size(values_.front()), iobs->original_length())) &&
        (mem_handles_.size() != 0 && mem_handles_.front() != IKVStore::HANDLE_NONE)) {

      /* Fast path: small size and memory already registered */
      CPLOG(1, "%s: using small send for direct put key=(%.*s) key_len=%lu value=(%.20s...) value_len=%lu", __func__, int(key_len_),
            static_cast<const char *>(key_), key_len_, static_cast<const char *>(::base(values_.front())), ::size(values_.front()));

      const auto msg =
        new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool_,
                                                              mcas::protocol::OP_PUT,  // op
                                                              key_, key_len_, size(values_.front()), flags_);

      if (_options.short_circuit_backend) msg->add_scbe();

      post_recv(&*iobr);

      iobs->set_length(msg->msg_len());
      iobs->iov[1].iov_base = const_cast<void*>(::base(values_.front()));
      iobs->iov[1].iov_len =  size(values_.front());
      iobs->desc[1] = static_cast<buffer_base *>(mem_handles_.front())->get_desc();
      post_send(iobs->iov, iobs->iov + 2, iobs->desc, &*iobs, msg, __func__); /* send two concatentated buffers in single DMA */

      out_async_handle_ = new async_buffer_set_simple(debug_level(), std::move(iobs), std::move(iobr));
    }
    else
    {
        /* for large puts, where the receiver will not have
         * sufficient buffer space, we use put locate (DMA write) protocol
         */
        out_async_handle_ = put_locate_async(TM_REF pool_, key_, key_len_, values_, rmd_,
                                             mem_handles_, flags_);
    }
    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const remote_fail &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return e.status();
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::async_get_direct(TM_ACTUAL const IMCAS::pool_t                        pool_,
                                              const void *const                          key_,
                                              const size_t                               key_len_,
                                              void *const                                value_,
                                              size_t &                                   value_len_,
                                              component::IMCAS::async_handle_t &         out_async_handle_,
                                              component::Registrar_memory_direct *       rmd_,
                                              const component::IKVStore::memory_handle_t mem_handle_,
                                              const component::IMCAS::flags_t            flags_)
{
  TM_SCOPE()
  API_LOCK();

  if (value_len_ && !value_) {
    PWRN("%s: bad parameter value=%p value_len=%zu", __func__, value_, value_len_);
    return E_BAD_PARAM;
  }

  assert(_max_message_size);

  if (pool_ == 0) {
    PWRN("%s: invalid pool identifier", __func__);
    return E_INVAL;
  }

  try {
    auto iobr = make_iob_ptr_recv();
    auto iobs = make_iob_ptr_send();

    out_async_handle_ = get_locate_async(TM_REF pool_, key_, key_len_, value_, value_len_, rmd_,
                                         mem_handle_ == IKVStore::HANDLE_NONE ? nullptr : static_cast<buffer_base *>(mem_handle_)->get_desc(), flags_);
    return S_OK;
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    return E_FAIL;
  }
  catch (const remote_fail &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return e.status();
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    return E_FAIL;
  }
}

status_t Connection_handler::check_async_completion(IMCAS::async_handle_t &handle)
{
  API_LOCK();
  auto bptrs = static_cast<async_buffer_set_t *>(handle);
  assert(bptrs);

  int status = E_BUSY;
  try
    {
      status = bptrs->move_along(this);
      /* status will be one of
       * E_BUSY: the operattion is not finished, but the call to move_along may have
       * caused progress other: the operation has finished
       */
    }
  catch ( const Exception &e )
    {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      status = E_FAIL;
    }
  catch ( const std::exception &e )
    {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      status = E_FAIL;
    }

  if (status != E_BUSY) {
    delete bptrs;
  }

  return status;
}

status_t Connection_handler::get(const pool_t pool, const std::string &key, std::string &value)
{
  TM_ROOT()
  API_LOCK();

  if (debug_level() > 1) {
    auto key_len = key.length();
    PINF("get: %.*s (key_len=%lu)", int(key_len), key.c_str(), key_len);
  }

  const auto iobs = make_iob_ptr_send();
  const auto iobr = make_iob_ptr_recv();
  assert(iobs);
  assert(iobr);

  status_t status;

  try {
    const auto msg =
      new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                            mcas::protocol::OP_GET,  // op
                                                            key, "", 0);

    if (_options.short_circuit_backend) msg->add_scbe();

    post_recv(&*iobr);
    sync_inject_send(&*iobs, msg, __func__);
    wait_for_completion(&*iobr);

    const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);
#if 0
    /* NOTE: g++ might or might not be smart enough to skip the string argument evaluation if the
     *  debug_level() check in msg_recv_log will suppress use of the string.
     * If it is not, performance will suffer.
     */
    msg_recv_log(response_msg, __func__ + std::string(" ") + std::string(response_msg->data(), response_msg->data_length()));
#endif
                 status = response_msg->get_status();
                 value.reserve(response_msg->data_length() + 1);
                 value.insert(0, response_msg->cdata(), response_msg->data_length());
                 assert(response_msg->data());
  }
  catch (const Exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
    status = E_FAIL;
  }
  catch (const std::exception &e) {
    PLOG("%s %s fail %s", __FILE__, __func__, e.what());
    status = E_FAIL;
  }

  return status;
}

  status_t Connection_handler::get(const pool_t pool, const std::string &key, void *&value, size_t &value_len)
  {
    TM_ROOT()
    API_LOCK();

    if (debug_level() > 1) {
      auto key_len = key.length();
      PINF("get: %.*s (key_len=%lu)", int(key_len), key.c_str(), key_len);
    }

    const auto iobs = make_iob_ptr_send();
    const auto iobr = make_iob_ptr_recv();
    assert(iobs);
    assert(iobr);

    status_t status;

    try {
      const auto msg =
        new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool,
                                                              mcas::protocol::OP_GET,  // op
                                                              key.c_str(), key.length(), 0);

      /* indicate how much space has been allocated on this side. For
         get this is based on buffer size
      */
      msg->set_availabe_val_len_from_iob_len(iobs->original_length());

      if (_options.short_circuit_backend) msg->add_scbe();

      post_recv(&*iobr);
      sync_inject_send(&*iobs, msg, __func__);
      {
        TM_SCOPE(wait_recv)
        wait_for_completion(&*iobr); /* TODO; could we issue the recv and send together? */
      }

      const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

      if (response_msg->get_status() != S_OK) return response_msg->get_status();

      CPLOG(1, "%s: message value (%.*s) size=%lu", __func__, int(response_msg->data_length()), response_msg->data(), response_msg->data_length());

      if (response_msg->is_set_twostage_bit()) {
        /* two-stage get */
        const auto data_len = response_msg->data_length() + 1;
        value               = ::aligned_alloc(MiB(2), data_len);
        if (value == nullptr) {
          throw std::bad_alloc();
        }
        madvise(value, data_len, MADV_HUGEPAGE);

        auto  region = make_memory_registered(common::make_const_byte_span(value, data_len)); /* we could have some pre-registered? */
        void *desc[] = {region.get_memory_descriptor()};

        ::iovec iov[]{{value, data_len - 1}};
        post_recv(std::begin(iov), std::end(iov), std::begin(desc), &iov[0]);

        /* synchronously wait for receive to complete */
        wait_for_completion(&iov);
        CPLOG(1, "%s Received value from two stage get", __func__);
      }
      else {
	TM_SCOPE(alloc)
        /* copy off value from IO buffer */
        value = ::malloc(response_msg->data_length() + 1);
        if (value == nullptr) {
          throw std::bad_alloc();
        }
	{
          TM_SCOPE(copy)
        value_len = response_msg->data_length();
        std::memcpy(value, response_msg->data(), response_msg->data_length());
        static_cast<char *>(value)[response_msg->data_length()] = '\0';
	}
      }

      status = response_msg->get_status();
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      status = E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      status = E_FAIL;
    }

    return status;
  }

  status_t Connection_handler::get_direct(const pool_t                              pool_,
                                          const void *const                         key_,
                                          const size_t                              key_len_,
                                          void *const                               value_,
                                          size_t &                                  value_len_,
                                          component::Registrar_memory_direct *const rmd_,
                                          const IMCAS::memory_handle_t              mem_handle_)
  {
    TM_ROOT()
    component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;

    auto status = async_get_direct(TM_REF pool_, key_, key_len_, value_, value_len_, async_handle, rmd_, mem_handle_, 0);
    if (status == S_OK) {
      TM_SCOPE(spin)
      do {
        status = check_async_completion(async_handle);
      } while (status == E_BUSY);
    }
    return status;
  }

  status_t Connection_handler::get_direct_offset(const pool_t                              pool_,
                                                 const std::size_t                         offset_,
                                                 std::size_t &                             length_,
                                                 void *const                               buffer_,
                                                 component::Registrar_memory_direct *const rmd_,
                                                 const component::IMCAS::memory_handle_t   mem_handle_)
  {
    TM_ROOT()
    component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;

    auto status = async_get_direct_offset(pool_, offset_, length_, buffer_, async_handle, rmd_, mem_handle_);
    if (status == S_OK) {
      TM_SCOPE(spin)
      do {
        status = check_async_completion(async_handle);
      } while (status == E_BUSY);
    }
    return status;
  }

  status_t Connection_handler::put_direct_offset(const pool_t                              pool_,
                                                 const std::size_t                         offset_,
                                                 std::size_t &                             size_,
                                                 const void *const                         buffer_,
                                                 component::Registrar_memory_direct *const rmd_,
                                                 const component::IMCAS::memory_handle_t   mem_handle_)
  {
    TM_ROOT()
    component::IMCAS::async_handle_t async_handle = component::IMCAS::ASYNC_HANDLE_INIT;

    auto status = async_put_direct_offset(pool_, offset_, size_, buffer_, async_handle, rmd_, mem_handle_);
    if (status == S_OK) {
      TM_SCOPE(spin)
      do {
        status = check_async_completion(async_handle);
      } while (status == E_BUSY);
    }
    return status;
  }

  status_t Connection_handler::async_get_direct_offset(const pool_t                              pool_,
                                                       const std::size_t                         offset_,
                                                       std::size_t &                             length_,
                                                       void *const                               buffer_,
                                                       IMCAS::async_handle_t &                   out_async_handle_,
                                                       component::Registrar_memory_direct *const rmd_,
                                                       const component::IMCAS::memory_handle_t   mem_handle_)
  {
    if(length_ == 0)
      throw API_exception("%s: variant of get_direct_offset called with zero length", __func__);

    API_LOCK();

    if (length_ && !buffer_) {
      PWRN("%s: bad parameter buffer=%p size=%zu", __func__, buffer_, length_);
      return E_BAD_PARAM;
    }

    try {
      out_async_handle_ = get_direct_offset_async(pool_, offset_, buffer_, length_, rmd_,
                                                  mem_handle_ == IMCAS::MEMORY_HANDLE_NONE ? nullptr : static_cast<buffer_base *>(mem_handle_)->get_desc());
      return S_OK;
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      return E_FAIL;
    }
    catch (const remote_fail &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      return e.status();
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      return E_FAIL;
    }
  }

  status_t Connection_handler::async_put_direct_offset(const pool_t                              pool_,
                                                       const std::size_t                         offset_,
                                                       std::size_t &                             length_,
                                                       const void *const                         buffer_,
                                                       IMCAS::async_handle_t &                   out_async_handle_,
                                                       component::Registrar_memory_direct *const rmd_,
                                                       const component::IMCAS::memory_handle_t   mem_handle_)
  {
    if(length_ == 0)
      throw API_exception("%s: variant of put_direct_offset called with zero length", __func__);

    API_LOCK();

    if (length_ && !buffer_) {
      PWRN("%s: bad parameter buffer=%p size=%zu", __func__, buffer_, length_);
      return E_BAD_PARAM;
    }

    try {
      out_async_handle_ = put_direct_offset_async(pool_, offset_, buffer_, length_, rmd_,
                                                  mem_handle_ == IMCAS::MEMORY_HANDLE_NONE ? nullptr : static_cast<buffer_base *>(mem_handle_)->get_desc());
      return S_OK;
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
    }
    return E_FAIL;
  }

  status_t Connection_handler::erase(const pool_t pool, const std::string &key)
  {
    API_LOCK();

    const auto iobs = make_iob_ptr_send();
    const auto iobr = make_iob_ptr_recv();
    assert(iobs);
    assert(iobr);

    status_t status;

    try {
      const auto msg = new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(), auth_id(), request_id(), pool, mcas::protocol::OP_ERASE, key.c_str(), key.length(), 0);

      post_recv(&*iobr);
      sync_inject_send(&*iobs, msg, __func__);
      wait_for_completion(&*iobr);

      const auto response_msg = msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, __func__);

      status = response_msg->get_status();
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      status = E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      status = E_FAIL;
    }

    return status;
  }

  status_t Connection_handler::async_erase(const IMCAS::pool_t    pool,
                                           const std::string &    key,
                                           IMCAS::async_handle_t &out_async_handle)
  {
    API_LOCK();

    auto iobs = make_iob_ptr_send();
    auto iobr = make_iob_ptr_recv();

    assert(iobs);
    assert(iobr);

    try {
      const auto msg = new (iobs->base()) mcas::protocol::Message_IO_request(iobs->length(),
                                                                             auth_id(),
                                                                             request_id(),
                                                                             pool,
                                                                             mcas::protocol::OP_ERASE,
                                                                             key.c_str(),
                                                                             key.length(),
                                                                             0);

      iobs->set_length(msg->msg_len());

      /* post both send and receive */
      post_recv(&*iobr);
      post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

      out_async_handle = new async_buffer_set_simple(debug_level(), std::move(iobs), std::move(iobr));
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      throw Logic_exception("%s: network posting failed unexpectedly.", __func__);
    }

    return S_OK;
  }

  size_t Connection_handler::count(const pool_t pool)
  {
    API_LOCK();

    const auto iobs = make_iob_ptr_send();
    const auto iobr = make_iob_ptr_recv();
    assert(iobs);
    assert(iobr);

    try {
      const auto msg =
        new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(), IKVStore::Attribute::COUNT, pool);

      post_recv(&*iobr);
      sync_inject_send(&*iobs, msg, msg->base_message_size(), __func__);
      wait_for_completion(&*iobr);

      const auto response_msg = msg_recv<const mcas::protocol::Message_INFO_response>(&*iobr, __func__);

      return response_msg->value();
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      return 0;
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      return 0;
    }
  }

  status_t Connection_handler::get_attribute(const IKVStore::pool_t    pool,
                                             const IKVStore::Attribute attr,
                                             std::vector<uint64_t> &   out_attr,
                                             const std::string *       key)
  {
    API_LOCK();

    const auto iobs = make_iob_ptr_send();
    const auto iobr = make_iob_ptr_recv();
    assert(iobs);
    assert(iobr);

    status_t status;

    try {
      const auto msg = new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(), attr, pool);

      if (key) msg->set_key(iobs->length(), *key);

      post_recv(&*iobr);
      sync_inject_send(&*iobs, msg, msg->message_size(), __func__);

      wait_for_completion(&*iobr);
      const auto response_msg = msg_recv<const mcas::protocol::Message_INFO_response>(&*iobr, __func__);

      out_attr.clear();
      out_attr.push_back(response_msg->value());
      status = response_msg->get_status();
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      status = E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      status = E_FAIL;
    }
    return status;
  }

  status_t Connection_handler::get_statistics(IMCAS::Shard_stats &out_stats)
  {
    API_LOCK();

    const auto iobs = make_iob_ptr_send();
    const auto iobr = make_iob_ptr_recv();
    assert(iobs);
    assert(iobr);

    status_t status;

    try {
      const auto msg =
        new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(), mcas::protocol::INFO_TYPE_GET_STATS, 0);

      post_recv(&*iobr);
      sync_inject_send(&*iobs, msg, msg->message_size(), __func__);

      wait_for_completion(&*iobr);
      const auto response_msg = msg_recv<const mcas::protocol::Message_stats>(&*iobr, __func__);

      status = response_msg->get_status();
#pragma GCC diagnostic push
#if defined __clang__ || 9 <= __GNUC__
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
#endif
      out_stats = response_msg->stats;
#pragma GCC diagnostic pop
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      status = E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      status = E_FAIL;
    }

    return status;
  }

  status_t Connection_handler::find(const IMCAS::pool_t pool,
                                    const std::string & key_expression,
                                    const offset_t      offset,
                                    offset_t &          out_matched_offset,
                                    std::string &       out_matched_key)
  {
    API_LOCK();

    const auto iobs = make_iob_ptr_send();
    const auto iobr = make_iob_ptr_recv();
    assert(iobs);
    assert(iobr);

    status_t status;

    try {
      const auto msg =
        new (iobs->base()) mcas::protocol::Message_INFO_request(auth_id(),
                                                                mcas::protocol::INFO_TYPE_FIND_KEY,
                                                                pool,
                                                                offset);

      msg->set_key(iobs->length(), key_expression);

      post_recv(&*iobr);
      sync_inject_send(&*iobs, msg, msg->message_size(), __func__);

      wait_for_completion(&*iobr);
      const auto response_msg = msg_recv<const mcas::protocol::Message_INFO_response>(&*iobr, "FIND");

      status = response_msg->get_status();

      if (status == S_OK) {
        out_matched_key    = response_msg->c_str();
        out_matched_offset = response_msg->Offset();
      }
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      status = E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      status = E_FAIL;
    }
    return status;
  }

  status_t Connection_handler::receive_and_process_ado_response(
    const iob_ptr & iobr_
    , std::vector<IMCAS::ADO_response> & out_response_
  )
  {
    const auto response_msg = msg_recv<const mcas::protocol::Message_ado_response>(&*iobr_, __func__);
    status_t status = response_msg->get_status();

    out_response_.clear();
    if (status == S_OK) {
      /* unmarshal responses */
      for (uint32_t i = 0; i < response_msg->get_response_count(); i++) {
        void *   out_data     = nullptr;
        size_t   out_data_len = 0;
        uint32_t out_layer_id = 0;
        response_msg->client_get_response(i, out_data, out_data_len, out_layer_id);

#if defined DEBUG_NPC_RESPONSES
        if (out_data_len > 0) {
          PLOG("%s: Response:", __func__);
          hexdump(out_data, out_data_len);
        }
        else {
          PLOG("%s: Response (inline): %p", __func__, out_data);
        }
#endif
        out_response_.emplace_back(out_data, out_data_len, out_layer_id);
      }
    }
    else {
      if (response_msg->get_response_count() > 0) {
        void *   err_msg      = nullptr;
        size_t   err_msg_len  = 0;
        uint32_t out_layer_id = 0;
        response_msg->client_get_response(0, err_msg, err_msg_len, out_layer_id);
        PLOG("%s:%u ADO response status %d %.*s", __FILE__, __LINE__, status, int(err_msg_len),
             static_cast<const char *>(err_msg));
        ::free(err_msg);
      }
    }
    return status;
  }

  template <typename MT>
    status_t Connection_handler::invoke_ado_common(
      const iob_ptr & iobs_
      , const MT *msg_
      , std::vector<IMCAS::ADO_response>& out_response_
      , unsigned int flags
    )
    {
      iobs_->set_length(msg_->message_size());

      if (flags & IMCAS::ADO_FLAG_ASYNC) {
        sync_send(&*iobs_, msg_, __func__);
        /* do not wait for response */
        return S_OK;
      }

      const auto iobr = make_iob_ptr_recv();
      assert(iobr);

      post_recv(&*iobr);
      sync_send(&*iobs_, msg_, __func__);
      wait_for_completion(&*iobr); /* wait for response */

      return receive_and_process_ado_response(iobr, out_response_);
    }

  status_t Connection_handler::invoke_ado(const IKVStore::pool_t            pool,
                                          const basic_string_view<byte>     key,
                                          const basic_string_view<byte>     request,
                                          const unsigned int                flags,
                                          std::vector<IMCAS::ADO_response>& out_response,
                                          const size_t                      value_size)
  {
    API_LOCK();

    const auto iobs = make_iob_ptr_send();
    assert(iobs);

    status_t status;

    try {
      const auto msg = new (iobs->base())
        mcas::protocol::Message_ado_request(iobs->length(),
                                            auth_id(),
                                            request_id(),
                                            pool,
                                            key,
                                            request,
                                            flags,
                                            value_size);
      status = invoke_ado_common(iobs, msg, out_response, flags);
    }
    catch (const Exception &e) {
      PLOG("%s:%u ADO response Exception %s", __FILE__, __LINE__, e.cause());
      status =  E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s:%u ADO response exception %s", __FILE__, __LINE__, e.what());
      status =  E_FAIL;
    }
    return status;
  }

  status_t Connection_handler::invoke_ado_async(const component::IMCAS::pool_t               pool,
                                                const basic_string_view<byte>                key,
                                                const basic_string_view<byte>                request,
                                                const component::IMCAS::ado_flags_t          flags,
                                                std::vector<component::IMCAS::ADO_response>& out_response,
                                                component::IMCAS::async_handle_t &           out_async_handle,
                                                const size_t                                 value_size)
  {
    API_LOCK();

    auto iobs = make_iob_ptr_send();
    auto iobr = make_iob_ptr_recv();

    assert(iobs);
    assert(iobr);

    try {
      const auto msg = new (iobs->base()) mcas::protocol::Message_ado_request(iobs->length(),
                                                                              auth_id(),
                                                                              request_id(),
                                                                              pool,
                                                                              key,
                                                                              request,
                                                                              flags,
                                                                              value_size);
      iobs->set_length(msg->message_size());

      post_recv(&*iobr);
      post_send(&*iobs, msg, __func__);

      out_async_handle = new async_buffer_set_invoke(debug_level(), std::move(iobs), std::move(iobr), &out_response);

      return S_OK;
    }
    catch (const Exception &e) {
      PLOG("%s:%u ADO response Exception %s", __FILE__, __LINE__, e.cause());
      return E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s:%u ADO response exception %s", __FILE__, __LINE__, e.what());
      return E_FAIL;
    }
    catch (...) {
      return E_FAIL;
    }
  }

  status_t Connection_handler::invoke_put_ado(const IKVStore::pool_t            pool,
                                              const basic_string_view<byte>     key,
                                              const basic_string_view<byte>     request,
                                              const basic_string_view<byte>     value,
                                              const size_t                      root_len,
                                              const unsigned int                flags,
                                              std::vector<IMCAS::ADO_response>& out_response)
  {
    API_LOCK();

    if (request.size() == 0) return E_INVAL;

    const auto iobs = make_iob_ptr_send();
    assert(iobs);

    status_t status;

    try {
      const auto msg = new (iobs->base()) mcas::protocol::Message_put_ado_request(iobs->length(),
                                                                                  auth_id(),
                                                                                  request_id(),
                                                                                  pool,
                                                                                  key,
                                                                                  request,
                                                                                  value,
                                                                                  root_len,
                                                                                  flags);

      status = invoke_ado_common(iobs, msg, out_response, flags);
    }
    catch (const Exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.cause());
      status = E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s %s fail %s", __FILE__, __func__, e.what());
      status = E_FAIL;
    }

    return status;
  }

  status_t Connection_handler::invoke_put_ado_async(const component::IMCAS::pool_t                  pool,
                                                    const basic_string_view<byte>                   key,
                                                    const basic_string_view<byte>                   request,
                                                    const basic_string_view<byte>                   value,
                                                    const size_t                                    root_len,
                                                    const component::IMCAS::ado_flags_t             flags,
                                                    std::vector<component::IMCAS::ADO_response>&    out_response,
                                                    component::IMCAS::async_handle_t&               out_async_handle)
  {
    API_LOCK();

    auto iobs = make_iob_ptr_send();
    auto iobr = make_iob_ptr_recv();

    assert(iobs);
    assert(iobr);

    try {
      const auto msg = new (iobs->base()) mcas::protocol::Message_put_ado_request(iobs->length(),
                                                                                  auth_id(),
                                                                                  request_id(),
                                                                                  pool,
                                                                                  key,
                                                                                  request,
                                                                                  value,
                                                                                  root_len,
                                                                                  flags);
      iobs->set_length(msg->message_size());

      post_recv(&*iobr);
      post_send(&*iobs, msg, __func__);

      out_async_handle = new async_buffer_set_invoke(debug_level(), std::move(iobs), std::move(iobr), &out_response);

      return S_OK;
    }
    catch (const Exception &e) {
      PLOG("%s:%u ADO response Exception %s", __FILE__, __LINE__, e.cause());
      return E_FAIL;
    }
    catch (const std::exception &e) {
      PLOG("%s:%u ADO response exception %s", __FILE__, __LINE__, e.what());
      return E_FAIL;
    }
    catch (...) {
      return E_FAIL;
    }

    return S_OK;
  }

  auto Connection_handler::make_iob_ptr(buffer_t::completion_t completion_) -> iob_ptr
  {
    return iob_ptr(allocate(completion_), this);
  }

  auto Connection_handler::make_iob_ptr_recv() -> iob_ptr
  {
    return make_iob_ptr(recv_complete);
  }

  auto Connection_handler::make_iob_ptr_send() -> iob_ptr
  {
    return make_iob_ptr(send_complete);
  }

  auto Connection_handler::make_iob_ptr_write() -> iob_ptr
  {
    return make_iob_ptr(write_complete);
  }

  auto Connection_handler::make_iob_ptr_read() -> iob_ptr
  {
    return make_iob_ptr(read_complete);
  }

  int Connection_handler::tick()
  {
    using namespace mcas::protocol;

    switch (_state) {
    case INITIALIZE: {
      set_state(HANDSHAKE_SEND);
      break;
    }
    case HANDSHAKE_SEND: {

      const auto iob = make_iob_ptr_send();
      auto       msg = new (iob->base()) mcas::protocol::Message_handshake(auth_id(), 1);
      msg->set_status(S_OK);

      /* set security options */
      msg->security_tls_auth = _options.tls;
      PNOTICE("TLS is %s", _options.tls ? "ON" : "OFF");
      iob->set_length(msg->msg_len());
      post_send(iob->iov, iob->iov + 1, iob->desc, &*iob, msg, __func__);

      try {
        wait_for_completion(&*iob);
      }
      catch (...) {
        PERR("%s %s handshake send failed", __FILE__, __func__);
        set_state(STOPPED);
      }

      static int sent = 0;
      sent++;
      PMAJOR(">>> Sent handshake (%d)", sent);

      set_state(HANDSHAKE_GET_RESPONSE);
      break;
    }
    case HANDSHAKE_GET_RESPONSE: {
      const auto iobr = make_iob_ptr_recv();
      post_recv(iobr->iov, iobr->iov + 1, iobr->desc, &*iobr);

      try {
        wait_for_completion(&*iobr);
        const auto response_msg = msg_recv<const mcas::protocol::Message_handshake_reply>(&*iobr, "handshake");

        /* server is indicating that it wants to start TLS session */
        if(response_msg->start_tls)
          start_tls();
      }
      catch (...) {
        PERR("%s %s handshake response failed", __FILE__, __func__);
        set_state(STOPPED);
      }

      static int recv = 0;
      recv++;

      set_state(READY);

      _max_message_size = max_message_size(); /* from fabric component */
      break;
    }
    case READY: {
      return 0;
      break;
    }
    case SHUTDOWN: {
      const auto iobs = make_iob_ptr_send();
      auto       msg  = new (iobs->base()) mcas::protocol::Message_close_session(reinterpret_cast<uint64_t>(this));

      iobs->set_length(msg->msg_len());
      post_send(iobs->iov, iobs->iov + 1, iobs->desc, &*iobs, msg, __func__);

      // server-side may have disappeared
      // try {
      //   wait_for_completion(&*iobs);
      // }
      // catch (...) {
      // }

      set_state(STOPPED);
      PLOG("Connection_handler::%s: connection %p shutdown.", __func__, common::p_fmt(this));
      return 0;
    }
    case STOPPED: {
      assert(0);
      return 0;
    }
    }  // end switch

    return 1;
  }

  unsigned TLS_transport::debug_level()
  {
    return TLS_DEBUG_LEVEL;
  }

  int TLS_transport::gnutls_pull_timeout_func(gnutls_transport_ptr_t, unsigned int ms)
  {
    return 0;
  }


  ssize_t TLS_transport::gnutls_pull_func(gnutls_transport_ptr_t connection, void* buffer, size_t buffer_size)
  {
    assert(connection);

    auto p_connection = reinterpret_cast<Connection_handler*>(connection);

    if(p_connection->_tls_buffer.remaining() >= buffer_size) {

      if(debug_level() > 2)
        PLOG("TLS pull: taking %lu bytes from remaining (%lu)", buffer_size, p_connection->_tls_buffer.remaining());

      p_connection->_tls_buffer.pull(buffer, buffer_size);
      return buffer_size;
    }

    auto iobr = p_connection->make_iob_ptr_recv();

    p_connection->post_recv(&*iobr);
    p_connection->wait_for_completion(&*iobr); /* await response */

    void * base_v = iobr->base();
    uint64_t * base = reinterpret_cast<uint64_t*>(base_v);
    uint64_t payload_size = base[0];

    if(debug_level() > 2)
      PLOG("TLS received: iob_len=%lu payload-len=%lu", iobr->length(), payload_size);

    p_connection->_tls_buffer.push(reinterpret_cast<void*>(&base[1]), payload_size);
    p_connection->_tls_buffer.pull(buffer, buffer_size);
    return buffer_size;
  }

  ssize_t TLS_transport::gnutls_vec_push_func(gnutls_transport_ptr_t connection, const giovec_t * iovec, int iovec_cnt )
  {
    assert(connection);
    auto p_connection = reinterpret_cast<Connection_handler*>(connection);
    auto iobs = p_connection->make_iob_ptr_send();

    void * base_v = iobs->base();
    uint64_t * base = reinterpret_cast<uint64_t*>(base_v);

    char * ptr = reinterpret_cast<char*>(&base[1]);
    size_t size = 0;

    for(int i=0; i<iovec_cnt; i++) {
      memcpy(ptr, iovec[i].iov_base, iovec[i].iov_len);
      size += iovec[i].iov_len;
      ptr += iovec[i].iov_len;
    }

    base[0] = size; /* prefix with length */
    iobs->set_length(size + sizeof(uint64_t));

    p_connection->sync_send(&*iobs, "TLS packet (client send)", __func__);

    if(debug_level() > 2)
      PLOG("TLS sent: %lu bytes (%p)", size, reinterpret_cast<void*>(&*iobs));

    return size;
  }

  void Connection_handler::start_tls()
  {
    if(_options.tls == false) throw Logic_exception("TLS contradiction");

    if(gnutls_global_init() != GNUTLS_E_SUCCESS)
      throw General_exception("gnutls_global_init() failed");

    if (gnutls_certificate_allocate_credentials(&_xcred) != GNUTLS_E_SUCCESS)
      throw General_exception("gnutls_certificate_allocate_credentials() failed");

    std::string cert_file(::getenv(ENVIRONMENT_VARIABLE_CERT));
    std::string key_file(::getenv(ENVIRONMENT_VARIABLE_KEY));

    PLOG("start_tls: cert=%s key=%s", cert_file.c_str(), key_file.c_str());

    if (gnutls_certificate_set_x509_key_file(_xcred, cert_file.c_str(), key_file.c_str(), GNUTLS_X509_FMT_PEM) !=
        GNUTLS_E_SUCCESS)
      throw General_exception("gnutls_certificate_set_x509_key_file() failed");

    if (gnutls_init(&_session, GNUTLS_CLIENT) != GNUTLS_E_SUCCESS)
      throw General_exception("gnutls_init() failed");

    if (gnutls_priority_init(&_priority, cipher_suite, NULL) != GNUTLS_E_SUCCESS)
      throw General_exception("gnutls_priority_init() failed");

    if (gnutls_priority_set(_session, _priority) != GNUTLS_E_SUCCESS)
      throw General_exception("gnutls_priority_set() failed");

    if (gnutls_credentials_set(_session, GNUTLS_CRD_CERTIFICATE, _xcred) != GNUTLS_E_SUCCESS)
      throw General_exception("gnutls_credentials_set() failed");

    //  gnutls_handshake_set_timeout(_session, GNUTLS_DEFAULT_HANDSHAKE_TIMEOUT);

    /* hook in TLS transport to use our RDMA connection */
    gnutls_transport_set_ptr(_session, this);
    gnutls_transport_set_vec_push_function(_session, TLS_transport::gnutls_vec_push_func);
    gnutls_transport_set_pull_function(_session, TLS_transport::gnutls_pull_func);
    gnutls_transport_set_pull_timeout_function(_session, TLS_transport::gnutls_pull_timeout_func);

    /* initiate handshake */
    int rc;
    if ((rc = gnutls_handshake(_session)) < 0) {
      if (rc == GNUTLS_E_CERTIFICATE_VERIFICATION_ERROR) {

        PERR("TLS certificate verification error");
        /* check certificate verification status */
        gnutls_datum_t out;
        auto           type   = gnutls_certificate_type_get(_session);
        auto           status = gnutls_session_get_verify_cert_status(_session);
        if (gnutls_certificate_verification_status_print(status, type, &out, 0) != GNUTLS_E_SUCCESS)
          throw General_exception("gnutls_certificate_verification_status_print() failed");

        gnutls_deinit(_session);
        gnutls_free(out.data);
      }
      throw General_exception("Client: handshake failed: %s\n", gnutls_strerror(rc));
    }

    /* get final result */
    gnutls_alert_description_t result;
    gnutls_record_recv(_session, &result, sizeof(result));

    if(result == GNUTLS_E_CERTIFICATE_VERIFICATION_ERROR)
      throw API_exception("server rejected certificate because verification failed");
    else if(result > 0)
      throw API_exception("TLS handshake rejected");

    PLOG("TLS handshake complete");
  }

}  // namespace client
}  // namespace mcas
