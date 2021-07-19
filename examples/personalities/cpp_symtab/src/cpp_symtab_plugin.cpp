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

#include "cpp_symtab_plugin.h"
#include <libpmem.h>
#include <api/interfaces.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <common/type_name.h>
#include <sstream>
#include <string>
#include <list>
#include <algorithm>
#include <ccpm/immutable_list.h>
#include <ccpm/immutable_string_table.h>
#include "cpp_symtab_types.h"
#include <btree_multimap.h>
#include <blindi_btree_hybrid_nodes_multimap.h>
#include <iostream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

using namespace symtab_ADO_protocol;
using namespace ccpm;
using namespace stx;
using namespace std;
using namespace boost;

#define END 16000

///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
struct GenericIndex {
public:
    virtual std::string name(void) = 0;
    virtual void PrintStat(void) {}
    virtual void UpdateThreadLocal(int num_threads) {}
    virtual void AssignGCID(int thread_id) {}
    virtual void UnregisterThread(int thread_id) {}
    virtual void PrintStats(void) {}
};



template<typename KeyType, typename ValueType>
struct Index : GenericIndex {
public:
    virtual bool Insert(const KeyType& key, const ValueType& value) = 0;
    virtual void GetValue(const KeyType &key,
                          std::vector<ValueType> &value_list)  = 0;

    virtual void GetRange(const KeyType &key,
                          std::vector<ValueType> &value_list)  = 0;


    virtual uint64_t GetRangeTime(const KeyType &key,
                  std::vector<ValueType> &value_list, uint64_t time_end) = 0;
};

struct UUID {
  uint64_t val[2];

public:
  bool operator<(const UUID& rhs) const {
    if (val[1] == rhs.val[1])
      return val[0] < rhs.val[0];
    else
      return val[1] < rhs.val[1];
  }

  bool operator==(const UUID& other) const {
    return val[0] == other.val[0] && val[1] == other.val[1];
  }
};


uint64_t char8B_to_uint64 (const char * input) { 
	uint64_t out;
        out = input[0]<<(8*7) | input[1]<<(8*6)
		| input[2]<<(8*5)
		| input[3]<<(8*4)
		| input[4]<<(8*3)
		| input[5]<<(8*2)
		| input[6]<<(8*1)
		| input[7];
	return out;
}



//////////////  STX tree      /////////////////////////////////////////

template<typename KeyType, typename ValueType, int InnerSlots, int LeafSlots>
struct BTreeType : Index<KeyType, ValueType> {
    btree_multimap<KeyType, ValueType, InnerSlots, LeafSlots> tree;

public:
    std::string name(void) { return std::string("BTree") + "-" + std::to_string(InnerSlots) + "-" + std::to_string(LeafSlots); }

    bool Insert(const KeyType& key, const ValueType& value) {
        tree.insert(key, value);
        return true;
    }

    void GetValue(const KeyType &key,
                  std::vector<ValueType> &value_list) {
        auto it = tree.lower_bound(key);
        if (it != tree.end())
            value_list.push_back(it->second);
    }

     void GetRange(const KeyType &key,
                  std::vector<ValueType> &value_list) {
        auto it = tree.lower_bound(key);
        for (int i=0; i < 3; i++) {
		if (it == tree.end())
			break;
		value_list.push_back(it->second);
		it++;
	}
    }
};

////////////////// Elastic /////////////////
template<typename KeyType, typename ValueType, template<typename, int> typename BlindiBtreeHybridType, int InnerSlots, int LeafSlots>
struct BlindiBTreeHybridNodes : Index<KeyType, ValueType> {
	    blindi_btree_hybrid_nodes_multimap<KeyType, ValueType, BlindiBtreeHybridType, InnerSlots, LeafSlots> tree;



std::string blindi_type;

public:
    BlindiBTreeHybridNodes(std::string type) : blindi_type(type) {}

    std::string name(void) { return blindi_type + "-" + std::to_string(InnerSlots) + "-" + std::to_string(LeafSlots); }

    bool Insert(const KeyType& key, const ValueType& value) {
        tree.insert(key, value);
     return true;
    }

    void GetValue(const KeyType &key,
                  std::vector<ValueType> &value_list) {

        auto it = tree.lower_bound(key);
	std::cout << " it.currslot " << it.currslot << std::endl;

        if (it.currslot != END) {
            value_list.push_back(it->second);
	}
	else {
		std::cout << " GetValue tree.end " << std::endl;
	}

    }

    void GetRange(const KeyType &key,
                  std::vector<ValueType> &value_list) {

        auto it = tree.lower_bound(key);

	for (int i=0; i < 10; i++) {
	std::cout << " it.currslot " << it.currslot << std::endl;
		if (it.currslot == ENDSLOT) {
			std::cout << " Range tree.end END= " << ENDSLOT << std::endl;
			break;
		}
		ValueType v1 = it->second; 
		std::cout << "value_range_ptr[0] = "<<  v1 << std::endl;
		std::cout << "value_range[0] = "<<  v1->val[0] << std::endl;
		std::cout << "value_range[1] = "<<  v1->val[1] << std::endl;
		std::cout << " it->second " << it->second << std::endl;
		value_list.push_back(it->second);
		it++;
	}

    }

    uint64_t GetRangeTime(const KeyType &key,
                  std::vector<ValueType> &value_list, uint64_t time_end) {

        auto it = tree.lower_bound(key);
	uint64_t cnt = 0;
        KeyType *curr_ts;
	std::cout << " current timestamp " << std::endl;
	while  (it.currslot != ENDSLOT) {
		std::cout << " current slot " << it.currslot << std::endl;
		curr_ts =  it->second;
		if (key.val[1] != curr_ts->val[1]){
			std::cout << "GetRangeTime the type are not identical key.type " << key.val[1] << " curr_ts " << curr_ts->val[1] << std::endl; 
			return cnt;
		}	
		if (curr_ts->val[0] > time_end){
			std::cout << "GetRangeTime the curr-time is above the time ; curr_ts " << curr_ts->val[0] << " time_end " << time_end << std::endl; 
			return cnt;
		}
		if (curr_ts->val[0] >= key.val[0]){
			std::cout << "GetRangeTime the curr-time in the right range count++ ; curr_ts " << curr_ts->val[0] << " key->val[0] " << key.val[0] << std::endl;
		       cnt++;	
		}
	       	else {
			std::cout << "GetRangeTime the curr-time is below  count stay the same ; curr_ts " << curr_ts->val[0] << " key->val[0] " << key.val[0] << std::endl; 
		}

		value_list.push_back(it->second);
		it++;
		std::cout << " current slot  after ++ " << it.currslot << std::endl;

	}
	return cnt;

    }

};
/////////////

const int INNER_NUM = 16;	
const int LEAF_NUM = 16;
//typedef string  KeyType;  
//typedef const char*  ValueType;  
typedef UUID  KeyType;  
typedef UUID*  ValueType;  
//Index<KeyType, ValueType> *idx = new BTreeType<KeyType, ValueType, INNER_NUM, LEAF_NUM>;
Index<KeyType, ValueType> *idx = new BlindiBTreeHybridNodes<KeyType, ValueType, SeqTreeBlindiNode, INNER_NUM, LEAF_NUM>("SeqTreeBlindi");



typedef  struct {
	uint64_t f0;
	uint64_t f1;
	uint64_t f2;
	uint64_t  f3;
	UUID  f4;
} item;

item table[5000]; 
uint64_t row_nu = 0;


///////////////////////////////////////////////////////


status_t ADO_symtab_plugin::register_mapped_memory(void * shard_vaddr,
                                                   void * local_vaddr,
                                                   size_t len) {
  PLOG("ADO_symtab_plugin: register_mapped_memory (%p, %p, %lu)", shard_vaddr,
       local_vaddr, len);

  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */
  return S_OK;
}

void ADO_symtab_plugin::launch_event(const uint64_t auth_id,
                                     const std::string& pool_name,
                                     const size_t pool_size,
                                     const unsigned int pool_flags,
                                     const unsigned int memory_type,
                                     const size_t expected_obj_count,
                                     const std::vector<std::string>& params)
{
}


status_t ADO_symtab_plugin::do_work(const uint64_t work_request_id,
                                    const char * key,
                                    size_t key_len,
                                    IADO_plugin::value_space_t& values,
                                    const void * in_work_request, /* don't use iovec because of non-const */
                                    const size_t in_work_request_len,
                                    bool new_root,
                                    response_buffer_vector_t& response_buffers)
{
  using namespace flatbuffers;
  using namespace symtab_ADO_protocol;
  using namespace cpp_symtab_personality;

  auto value = values[0].ptr;

  constexpr size_t buffer_increment = 32*1024; /* granularity for memory expansion */

  //  PLOG("invoke: value=%p value_len=%lu", value, value_len);

  std::cout << "root " << std::endl;
  auto& root = *(new (value) Value_root);
  bool force_init = false;

  if(root.num_regions == 0) {
    void * buffer;
    if(cb_allocate_pool_memory(buffer_increment, 8, buffer)!=S_OK)
      throw std::runtime_error("unable to allocate new region");
    root.add_region(buffer, buffer_increment);
    force_init = true;
  }


  Verifier verifier(static_cast<const uint8_t*>(in_work_request), in_work_request_len);
  if(!VerifyMessageBuffer(verifier)) {
    PMAJOR("unknown command flatbuffer");
    return E_INVAL;
  }

  auto msg = GetMessage(in_work_request);

  /*----------------------*/
  /* put request handling */
  /*----------------------*/
  auto put_request = msg->command_as_PutRequest();
  if(put_request) {
    auto str = put_request->word()->c_str();

  const char *p;  


// STX
    std::cout << "devide fields " << std::endl;
   std::vector<string> fields;
   boost::split( fields, str, is_any_of(" ") );
   uint64_t num0 = lexical_cast<uint64_t>(fields[0]);

   const char * c = fields[1].c_str();
   uint64_t num1 = char8B_to_uint64(c);

   std::stringstream ss;
   uint64_t num2;
   ss << std::hex << fields[2];
   ss >> num2;
   fields[3].erase(fields[3].length()-1);
   uint64_t num3 = lexical_cast<uint64_t>(fields[3]);
   std::cout << "num0 " << num0 << std::endl; 
   std::cout << "num1 " << num1 << "  felids1 " << fields[1].c_str() <<  std::endl; 
   std::cout << "num2 " << num2 << "  felids2 " << fields[2].c_str() << std::endl; 
   std::cout << "num3 " << num3 << std::endl; 
///   table[row_nu] = {num0, num1, num2, num3};
   table[row_nu] = {num0, num1, num2, num3};
   table[row_nu].f4.val[0] = num0;
   table[row_nu].f4.val[1] = num1;
//   idx->Insert(table[row_nu].f0.c_str(), table[row_nu].f0.c_str());
   std::cout << "idx->Insert"   << std::endl;
   idx->Insert(table[row_nu].f4, &table[row_nu].f4);

   PLOG("fields[0] = %s", fields[0].c_str());
   PLOG("fields[1] = %s", fields[1].c_str());
   PLOG("fields[2] = %s", fields[2].c_str());
   PLOG("fields[3] = %s", fields[3].c_str());
   std::vector<ValueType> v {};
   idx->GetValue(table[row_nu].f4, v);
   std::cout << "end row_nu " << row_nu << std::endl;
   std::cout << "valuei_ptr = " << v[0] << std::endl;
   std::cout << "value[0] = " << v[0]->val[0] << std::endl;
   std::cout << "value[1] = " << v[0]->val[1] << std::endl;
   if (row_nu > 0){
	   std::vector<ValueType> v1 {};
           idx->GetRange((table[row_nu].f4), v1);
	   std::cout << "value_range_ptr[0] = "<<  v1[0] << std::endl;
	   std::cout << "value_range[0] = "<<  v1[0]->val[0] << std::endl;
	   std::cout << "value_range[1] = "<<  v1[0]->val[1] << std::endl;
           uint64_t cnter = idx->GetRangeTime((table[0].f4), v1, 1361306);
	   std::cout << " the COUNTER in this range is "<< cnter << std::endl;
//	   std::cout << "value_range_ptr[1] = "<<  v1[1] << std::endl;
//	   std::cout << "value_range[1][0] = "<<  v1[1]->val[0] << std::endl;
//	   std::cout << "value_range_ptr[1] = "<<  v1[1] << std::endl;
//	   std::cout << "value_range[1][0] = "<<  v1[1]->val[0] << std::endl;
//	   std::cout << "value_range[1][1] = "<<  v1[1]->val[1] << std::endl;
   }
   row_nu++;
   return S_OK;
  }

  /*------------------------------*/
  /* build index request handling */
  /*------------------------------*/
  if(msg->command_as_BuildIndex()) {
    return S_OK;
  }

  /*---------------------------------------*/
  /* requesting symbol for given string    */
  /*---------------------------------------*/  
  auto get_symbol_request = msg->command_as_GetSymbol();
  if(get_symbol_request) {
    char *str1 = "string Literal";
    auto& req_word = *(get_symbol_request->word());
    std::vector<ValueType> v2 {};
    uint64_t cnter = idx->GetRangeTime((table[0].f4), v2, 1361306);
    std::cout << " the COUNTER symbol in this range is "<< cnter << std::endl;
    auto result = new uint64_t;
    *result = reinterpret_cast<uint64_t>(str1);
    response_buffers.emplace_back(result, sizeof(uint64_t), response_buffer_t::alloc_type_malloc{});
    return S_OK;
  }

  /*---------------------------------------*/
  /* requesting string for provided sym id */
  /*---------------------------------------*/
  auto get_string_request = msg->command_as_GetString();
  if(get_string_request) { 
    return S_OK;
  }

  PERR("unhandled command");
  return E_INVAL;
}

status_t ADO_symtab_plugin::shutdown() {
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void *factory_createInstance(component::uuid_t interface_iid) {
  PLOG("instantiating cpp-symtab-plugin");
  if (interface_iid == interface::ado_plugin)
    return static_cast<void *>(new ADO_symtab_plugin());
  else
    return NULL;
}

#undef RESET_STATE


