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

#include "cw_fabric_test.h"

#include <api/components.h>
#include <api/fabric_itf.h> /* IFabric, IFabric_factory */

#include <common/env.h>
#include <common/json.h>

#include <gsl/pointers>

#include <cstddef> /* size_t */
#include <cstdint> /* uint16_t, uint64_t */
#include <iostream> /* cerr */
#include <memory> /* make_shared, shared_ptr */

#if 0
struct fabric_component
{
private:
	gsl::not_null<component::IBase *> _comp;

public:
	fabric_component()
		: _comp( component::load_component( "libcomponent-fabric.so" , component::net_fabric_factory))
	{}

	gsl::not_null<component::IBase *> comp() const { return _comp; }
};

struct fabric_factory
{
	component::Itf_ref<component::IFabric_factory> _factory;
	const component::Itf_ref<component::IFabric_factory> & factory() const { return _factory; }
public:
	fabric_factory(fabric_component &c_)
		: _factory(component::make_itf_ref(static_cast<component::IFabric_factory *>(c_.comp()->query_interface(component::IFabric_factory::iid()))))
	{}
};

struct fabric_fabric
{
	static const std::uint16_t control_port;
	static const std::size_t memory_size;
	static const std::size_t data_size;
private:
	static std::string fabric_spec() {
		namespace c_json = common::json;
		using json = c_json::serializer<c_json::dummy_writer>;

		auto mr_mode =
			json::array(
				"FI_MR_LOCAL"
				, "FI_MR_VIRT_ADDR"
				, "FI_MR_ALLOCATED"
				, "FI_MR_PROV_KEY"
			);

		auto domain_name_spec = json::object();

		auto fabric_attr =
			json::member
			( "fabric_attr"
				, json::object(json::member("prov_name", json::string("verbs")))
			);
		auto domain_attr =
			json::member(
				"domain_attr"
				, std::move(
						json::object(
							json::member("mr_mode", std::move(mr_mode))
							, json::member("threading", "FI_THREAD_SAFE")
						).append(std::move(domain_name_spec))
					)
			);
		auto ep_attr =
			json::member(
				"ep_attr"
				, json::object(json::member("type", "FI_EP_MSG"))
			);
		return json::object(std::move(fabric_attr), std::move(domain_attr), std::move(ep_attr)).str();
	}

	std::unique_ptr<component::IFabric> _fabric;
public:
	fabric_fabric(fabric_factory &f_)
		: _fabric(f_.factory()->make_fabric(fabric_spec()))
	{}

	component::IFabric *fabric() const { return _fabric.get(); }
};
#endif
const std::uint16_t fabric_fabric::control_port = common::env_value<std::uint16_t>("FABRIC_TEST_CONTROL_PORT", 47591);
const std::size_t fabric_fabric::data_size = common::env_value<std::size_t>("SIZE", 1U<<23);
const std::size_t fabric_fabric::memory_size = fabric_fabric::data_size + 100;

#if 0
namespace
{
	auto empty_object_json = common::json::serializer<common::json::dummy_writer>::object{};
}
#endif
