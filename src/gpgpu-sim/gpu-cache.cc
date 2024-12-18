// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington,
// Vijay Kandiah, Nikos Hardavellas, Mahmoud Khairy, Junrui Pan,
// Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-cache.h"
#include "gpu-sim.h"
#include "hashing.h"
#include "stat-tool.h"
#include <assert.h>

// used to allocate memory that is large enough to adapt the changes in cache
// size across kernels

const char *cache_request_status_str(enum cache_request_status status) {
    static const char *static_cache_request_status_str[] = {
        "HIT",         "HIT_RESERVED", "MISS", "RESERVATION_FAIL",
        "SECTOR_MISS", "MSHR_HIT"};

    assert(sizeof(static_cache_request_status_str) / sizeof(const char *) ==
           NUM_CACHE_REQUEST_STATUS);
    assert(status < NUM_CACHE_REQUEST_STATUS);

    return static_cache_request_status_str[status];
}

const char *cache_fail_status_str(enum cache_reservation_fail_reason status) {
    static const char *static_cache_reservation_fail_reason_str[] = {
        "LINE_ALLOC_FAIL", "MISS_QUEUE_FULL", "MSHR_ENRTY_FAIL",
        "MSHR_MERGE_ENRTY_FAIL", "MSHR_RW_PENDING"};

    assert(sizeof(static_cache_reservation_fail_reason_str) /
               sizeof(const char *) ==
           NUM_CACHE_RESERVATION_FAIL_STATUS);
    assert(status < NUM_CACHE_RESERVATION_FAIL_STATUS);

    return static_cache_reservation_fail_reason_str[status];
}

unsigned l1d_cache_config::set_bank(new_addr_type addr) const {
    // For sector cache, we select one sector per bank (sector interleaving)
    // This is what was found in Volta (one sector per bank, sector
    // interleaving) otherwise, line interleaving
    return cache_config::hash_function(
        addr, l1_banks, l1_banks_byte_interleaving_log2, l1_banks_log2,
        l1_banks_hashing_function);
}

unsigned cache_config::set_index(new_addr_type addr) const {
    return cache_config::hash_function(addr, m_nset, m_line_sz_log2,
                                       m_nset_log2, m_set_index_function);
}

unsigned cache_config::hash_function(new_addr_type addr, unsigned m_nset,
                                     unsigned m_line_sz_log2,
                                     unsigned m_nset_log2,
                                     unsigned m_index_function) const {
    unsigned set_index = 0;

    switch (m_index_function) {
    case FERMI_HASH_SET_FUNCTION: {
        /*
         * Set Indexing function from "A Detailed GPU Cache Model Based on Reuse
         * Distance Theory" Cedric Nugteren et al. HPCA 2014
         */
        unsigned lower_xor = 0;
        unsigned upper_xor = 0;

        if (m_nset == 32 || m_nset == 64) {
            // Lower xor value is bits 7-11
            lower_xor = (addr >> m_line_sz_log2) & 0x1F;

            // Upper xor value is bits 13, 14, 15, 17, and 19
            upper_xor = (addr & 0xE000) >> 13;   // Bits 13, 14, 15
            upper_xor |= (addr & 0x20000) >> 14; // Bit 17
            upper_xor |= (addr & 0x80000) >> 15; // Bit 19

            set_index = (lower_xor ^ upper_xor);

            // 48KB cache prepends the set_index with bit 12
            if (m_nset == 64)
                set_index |= (addr & 0x1000) >> 7;

        } else { /* Else incorrect number of sets for the hashing function */
            assert("\nGPGPU-Sim cache configuration error: The number of sets "
                   "should "
                   "be "
                   "32 or 64 for the hashing set index function.\n" &&
                   0);
        }
        break;
    }

    case BITWISE_XORING_FUNCTION: {
        new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
        unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
        set_index = bitwise_hash_function(higher_bits, index, m_nset);
        break;
    }
    case HASH_IPOLY_FUNCTION: {
        new_addr_type higher_bits = addr >> (m_line_sz_log2 + m_nset_log2);
        unsigned index = (addr >> m_line_sz_log2) & (m_nset - 1);
        set_index = ipoly_hash_function(higher_bits, index, m_nset);
        break;
    }
    case CUSTOM_SET_FUNCTION: {
        /* No custom set function implemented */
        break;
    }

    case LINEAR_SET_FUNCTION: {
        set_index = (addr >> m_line_sz_log2) & (m_nset - 1);
        break;
    }

    default: {
        assert("\nUndefined set index function.\n" && 0);
        break;
    }
    }

    // Linear function selected or custom set index function not implemented
    assert((set_index < m_nset) &&
           "\nError: Set index out of bounds. This is caused by "
           "an incorrect or unimplemented custom set index function.\n");

    return set_index;
}

void l2_cache_config::init(linear_to_raw_address_translation *address_mapping) {
    cache_config::init(m_config_string, FuncCachePreferNone);
    m_address_mapping = address_mapping;
}

unsigned l2_cache_config::set_index(new_addr_type addr) const {
    new_addr_type part_addr = addr;

    if (m_address_mapping) {
        // Calculate set index without memory partition bits to reduce set
        // camping
        part_addr = m_address_mapping->partition_address(addr);
    }

    return cache_config::set_index(part_addr);
}

tag_array::~tag_array() {
    unsigned cache_lines_num = m_config.get_max_num_lines();
    for (unsigned i = 0; i < cache_lines_num; ++i)
        delete m_lines[i];
    delete[] m_lines;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id,
                     cache_block_t **new_lines)
    : m_config(config), m_lines(new_lines) {
    init(core_id, type_id);
}

void tag_array::update_cache_parameters(cache_config &config) {
    m_config = config;
}

tag_array::tag_array(cache_config &config, int core_id, int type_id)
    : m_config(config) {
    // assert( m_config.m_write_policy == READ_ONLY ); Old assert
    unsigned cache_lines_num = config.get_max_num_lines();
    m_lines = new cache_block_t *[cache_lines_num];
    if (config.m_cache_type == NORMAL) {
        for (unsigned i = 0; i < cache_lines_num; ++i)
            m_lines[i] = new line_cache_block();
    } else if (config.m_cache_type == SECTOR) {
        for (unsigned i = 0; i < cache_lines_num; ++i)
            m_lines[i] = new sector_cache_block();
    } else
        assert(0);

    init(core_id, type_id);
}

void tag_array::init(int core_id, int type_id) {
    m_access = 0;
    m_miss = 0;
    m_pending_hit = 0;
    m_res_fail = 0;
    m_sector_miss = 0;
    // initialize snapshot counters for visualizer
    m_prev_snapshot_access = 0;
    m_prev_snapshot_miss = 0;
    m_prev_snapshot_pending_hit = 0;
    m_core_id = core_id;
    m_type_id = type_id;
    is_used = false;
    m_dirty = 0;
}

void tag_array::add_pending_line(mem_fetch *mf) {
    assert(mf);
    new_addr_type addr = m_config.block_addr(mf->get_addr());
    line_table::const_iterator i = pending_lines.find(addr);
    if (i == pending_lines.end()) {
        pending_lines[addr] = mf->get_inst().get_uid();
    }
}

void tag_array::remove_pending_line(mem_fetch *mf) {
    assert(mf);
    new_addr_type addr = m_config.block_addr(mf->get_addr());
    line_table::const_iterator i = pending_lines.find(addr);
    if (i != pending_lines.end()) {
        pending_lines.erase(addr);
    }
}

/**
 * probe()函数的另一种封装模式
 */
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_fetch *mf, bool is_write,
                                           bool probe_mode) const {
    mem_access_sector_mask_t mask = mf->get_access_sector_mask();
    return probe(addr, idx, mask, is_write, probe_mode, mf);
}

/**
 * probe()函数用于在不改变cache block的状态(不影响LRU)检查addr是否命中，并检查cache能否处理该请求
 * 如果tag匹配，该cache line要么直接返回data，要么等待data fill，都用index记录对应的way id
 * 如果tag不匹配，那么判断当前cache line能否被驱逐作为data fill的新cache line，用index记录驱逐的way id
 * 但是只要后面的tag匹配，依然优先返回hit，如果所有的way都不匹配，返回RESERVATION_FAIL(表示无法处理)/MISS(表示未命中但是可以处理)
    * new_addr_type addr：输入地址addr是cache block的地址
    * unsigned &idx：如果cache tag匹配，用来记录对应的set index；如果tag不匹配，记录需要驱逐的cache line的way index；
    * mem_access_sector_mask_t mask：通过line->get_status(mask)得到sector的所有状态
    * bool is_write：表示是否是write请求
    * bool probe_mode：暂时为无效参数
    * mem_fetch *mf：暂时为无效参数
 */
enum cache_request_status tag_array::probe(new_addr_type addr, unsigned &idx,
                                           mem_access_sector_mask_t mask,
                                           bool is_write, bool probe_mode,
                                           mem_fetch *mf) const {
    // assert( m_config.m_write_policy == READ_ONLY );

    /*m_config.set_index(addr) 是返回一个地址 addr 在 Cache 中的 set index。这里的 set index 有一整套的映射函数*/
    unsigned set_index = m_config.set_index(addr);

    /*tag 返回addr对应的标签*/
    new_addr_type tag = m_config.tag(addr);

    /*如果存在invalid的cache line，优先驱逐，invalid_line记录 */
    unsigned invalid_line = (unsigned)-1;

    /*如果没有invalid，则从valid中使用LRU选择，valid_line记录 */
    unsigned valid_line = (unsigned)-1;

    /*LRU策略使用 */
    unsigned long long valid_timestamp = (unsigned)-1;

    /*all_reserved 被初始化为 true，是指所有 cache line都没有能够逐出来为新访问提供 RESERVE 的空间，标记cache 是否可以处理该request*/
    bool all_reserved = true;

    /*遍历该set_index对应的set上的所有way，通过tag判断是否匹配*/
    for (unsigned way = 0; way < m_config.m_assoc; way++) {
        /*m_assoc是一个set有多少way，二维展开为一维，m_config.m_assoc表示一个set中有多少way*/
        unsigned index = set_index * m_config.m_assoc + way;
        cache_block_t *line = m_lines[index];
        /*Tag相符。m_tag和tag均是：{除offset位以外的所有位, offset'b0}*/
        if (line->m_tag == tag) {
            if (line->get_status(mask) == RESERVED) {
                /**
                 * RESERVED表示之前的cache miss选择该cache block被选为填充新的数据，
                 * 已经向lower memory发送fill request了，主要处理位于同一cache line的连续地址访问
                */
                idx = index;
                return HIT_RESERVED;
            } else if (line->get_status(mask) == VALID) {
                /*VALID说明当前cache line可以立即返回数据 */
                idx = index;
                return HIT;
            } else if (line->get_status(mask) == MODIFIED) {
                /**
                 * L1 cache与L2 cache write hit时，采用write-back策略，只将数据写入该block，并不直接更新下级存储，
                 * 只有当这个块被替换时，才将数据写回下级存储
                 * MODIFIED，说明该 block 或 sector的数据已经被其他线程修改。
                 * 如果当前访问也是写操作的话即为写命中，直接覆盖写即可，即write hit，不需要考虑一致性的问题。
                 * When a sector read request is received to a modified sector, 
                 * it first checks if the sector write-mask is complete, i.e. 
                 * all the bytes have been written to and the line is fully readable
                 * If so, it reads the sector, otherwise, similar to fetch-on-write, it generates a read request for this sector 
                 * and merges it with the modified bytes
                */
                if ((!is_write && line->is_readable(mask)) || is_write) {
                    idx = index;
                    return HIT;
                } else {
                    idx = index;
                    return SECTOR_MISS;
                }

            } 
            /**
             * 下面的分支用于处理sector cache，即line为INVALID
             * 那么对于line cache，line->is_valid_line()即返回false
             * 但是对于sector cache，只要有一个sector不为INVALID即返回true
             * 判断为：当前sector为invalid，但是sector所处的line含有不是invalid的sector
            */
            else if (line->is_valid_line() &&
                       line->get_status(mask) == INVALID) {
                idx = index;
                return SECTOR_MISS;
            } else {
                assert(line->get_status(mask) == INVALID);
            }
        }

        /**
         * 上面分支如果不返回则说明line->m_tag != tag，
         * 对于该set的所有的cache block只要满足line->m_tag != tag，即判断能否作为被驱逐的cache block
         */

        /*首先被驱逐的cache line不能是reserved，因为其正在填充数据 */
        if (!line->is_reserved_line()) {
            /*dirty lines在整个cache line中的百分比*/
            float dirty_line_percentage =
                ((float)m_dirty / (m_config.m_nset * m_config.m_assoc)) * 100;
            /*优先不驱逐状态为MODIFIED的cache line，因为会造成额外的写回流量，但是如果整个dirty 的 cache line 的比例超过 m_wr_percent（V100，
            ，说明dirty data太多，也可以选择驱逐MODIFIED  */
            if (!line->is_modified_line() ||
                dirty_line_percentage >= m_config.m_wr_percent) {
                /*说明存在可以驱逐的cache line */
                all_reserved = false;
                /*如果整个cache line状态为invalid，则优先于LRU等策略，直接驱逐，节省带宽 */
                if (line->is_invalid_line()) {
                    invalid_line = index;
                } else {
                    /**LRU驱逐策略：
                     * 1. get_last_access_time()获取上次访问的时间戳,越小说明距离上次访问的时间越长
                     * 2. valid_timestamp 越小说明最近访问
                     * 3. valid_timestamp 被初始化为 (unsigned)-1，即可以看作无穷大，代表最优的set index预计的时间戳
                     */
                    if (m_config.m_replacement_policy == LRU) {
                        if (line->get_last_access_time() < valid_timestamp) {
                            valid_timestamp = line->get_last_access_time();
                            valid_line = index;
                        }
                    } 
                    /**
                     * FIFO驱逐策略：
                     * 1. get_alloc_time()越小说明,分配的越早
                     * 2. 越早越优先驱逐
                     */
                    else if (m_config.m_replacement_policy == FIFO) {
                        if (line->get_alloc_time() < valid_timestamp) {
                            valid_timestamp = line->get_alloc_time();
                            valid_line = index;
                        }
                    }
                }
            }
        }
    }

    /*下面是for循环遍历完所有的way结束后的代码， all_reserved为true说明该set不存在可以驱逐的cache line， RESERVATION_FAIL表示没有资源处理cache miss*/
    if (all_reserved) {
        assert(m_config.m_alloc_policy == ON_MISS);
        return RESERVATION_FAIL; 
    }

    /**
     * 如果tag匹配，idx记录set index
     * 如果tag不匹配使用idx记录需要驱逐的cache line所在的way，优先驱逐全部为invalid的cache line
    */
    if (invalid_line != (unsigned)-1) {
        idx = invalid_line;
    } else if (valid_line != (unsigned)-1) {
        idx = valid_line;
    } else
        abort(); 

    /*最后返回MISS，说明tag不命中，但是可以分配出资源处理cache miss，并向下级发送fill request */
    return MISS;
}

enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, mem_fetch *mf) {
    bool wb = false;
    evicted_block_info evicted;
    enum cache_request_status result = access(addr, time, idx, wb, evicted, mf);
    assert(!wb);
    return result;
}

/**
 * tag_array::access()在实际访问cache会调用，会修改tag_array对应的统计数据
 * tag_array::access()会实际修改LRU stack和对应的line的状态，并且影响cache访问统计数据
 * 其首先使用probe()检查地址addr对应的cache line的状态：
 *    - 对于HIT/HIT_RESERVED等tag匹配，修改LRU状态和统计统计数据
 *    - 对于MISS，需要根据on_miss/on_fill去分配不同资源处理
 *    - 对于RESERVATION_FAIL，说明该cache无法处理当前request
 * 
 * wb：本次access()是否造成向下一级内存写回数据
 * evicted_block_info &evicted：用于设置被驱逐的cache line的信息
 * 
 */
enum cache_request_status tag_array::access(new_addr_type addr, unsigned time,
                                            unsigned &idx, bool &wb,
                                            evicted_block_info &evicted,
                                            mem_fetch *mf) {
    /*对当前cache的访问次数+1*/
    m_access++;
    /*标记当前 tag_array 所属 cache 是否被使用过。一旦有 access() 函数被调用，则说明被使用过。*/
    is_used = true;
    /*log accesses to cache*/
    shader_cache_access_log(m_core_id, m_type_id, 0); 
    /*使用probe()判断如何处理该request*/
    enum cache_request_status status = probe(addr, idx, mf, mf->is_write());
    switch (status) {
    case HIT_RESERVED:
        /*HIT_RESERVED 的话，表示当前cache的数据块正在回填，不需要向下级memory发送请求，只需要等待data response*/
        m_pending_hit++;
    case HIT:
        /*如果HIT的话，设置第 idx 号 cache line 以及 mask 对应的 sector最后访问为参数time */
        m_lines[idx]->set_last_access_time(time, mf->get_access_sector_mask());
        break;
    case MISS:
        /*MISS说明已经选定 m_lines[idx] 作为逐出并 reserve 新访问的空间*/
        m_miss++;
        /*log cache misses*/ 
        shader_cache_access_log(m_core_id, m_type_id, 1);
        /**默认cache line的分配策略为allocate-on-miss，即处理miss时需要分配：
         * - a cache line slot，其中被分配的cache line使用idx标记
         * - a mshr entry
         * - a miss queue entry
         */
        if (m_config.m_alloc_policy == ON_MISS) {
            /*idx对应的cache line数据是dirty，立即写回下一级内存体系 */
            if (m_lines[idx]->is_modified_line()) {
                wb = true;
                // m_lines[idx]->set_byte_mask(mf);
                /*设置被逐出 cache line 的信息*/
                evicted.set_info(m_lines[idx]->m_block_addr,
                                 m_lines[idx]->get_modified_size(),
                                 m_lines[idx]->get_dirty_byte_mask(),
                                 m_lines[idx]->get_dirty_sector_mask());
                /*由于执行写回操作，MODIFIED 造成的 m_dirty 数量应该减1*/
                m_dirty--;
            }

            /*把需要reserve的cache line设置为新的tag，状态设置为RESERVED*/
            m_lines[idx]->allocate(m_config.tag(addr),
                                   m_config.block_addr(addr), time,
                                   mf->get_access_sector_mask());
        }
        break;
    case SECTOR_MISS:
        /*SECTOR_MISS表示Cache block 有效，但是其中的 byte mask = Cache block[mask] 状态无效*/
        assert(m_config.m_cache_type == SECTOR);
        m_sector_miss++;
        /*log cache misses */
        shader_cache_access_log(m_core_id, m_type_id, 1); 

        /* */
        if (m_config.m_alloc_policy == ON_MISS) {
            /*该line有sector状态为modified*/
            bool before = m_lines[idx]->is_modified_line();
            /*把对应sector的状态改为RESERVED */
            ((sector_cache_block *)m_lines[idx])
                ->allocate_sector(time, mf->get_access_sector_mask());
            /*修改完后该line所有的sector状态都不为modified */
            if (before && !m_lines[idx]->is_modified_line()) {
                m_dirty--;
            }
        }
        break;
    case RESERVATION_FAIL:
        /*当前cache不能分配更多的资源比如mshr去处理该miss*/
        m_res_fail++;
        /*log cache misses */
        shader_cache_access_log(m_core_id, m_type_id, 1); 
        break;
    default:
        fprintf(stderr,
                "tag_array::access - Error: Unknown"
                "cache_request_status %d\n",
                status);
        abort();
    }
    return status;
}

/**
 * 采用ON_FILL策略
 */
void tag_array::fill(new_addr_type addr, unsigned time, mem_fetch *mf,
                     bool is_write) {
    fill(addr, time, mf->get_access_sector_mask(), mf->get_access_byte_mask(),
         is_write);
}

/**
 * ON_FILL：采取的填充策略，需要在data返回时现场选一个 victim cache line进行驱逐
 * tag_array::fill()用来自low level的内存响应来填充cache line，更新cache line状态
 * 
*/
void tag_array::fill(new_addr_type addr, unsigned time,
                     mem_access_sector_mask_t mask,
                     mem_access_byte_mask_t byte_mask, bool is_write) {
    // assert( m_config.m_alloc_policy == ON_FILL );
    unsigned idx;

    /*使用probe()选择idx的cache line进行驱逐*/
    enum cache_request_status status = probe(addr, idx, mask, is_write);

    /*无法驱逐任何一个cache line*/
    if (status == RESERVATION_FAIL) {
        return;
    }

    /*检查需要驱逐的的当前cache line是否是有dirty data */
    bool before = m_lines[idx]->is_modified_line();
    // assert(status==MISS||status==SECTOR_MISS); // MSHR should have prevented
    // redundant memory request
    if (status == MISS) {
        /*修改当前cache block对应的tag等，状态修改为RESERVED表示等待数据回填*/
        m_lines[idx]->allocate(m_config.tag(addr), m_config.block_addr(addr),
                               time, mask);
    } else if (status == SECTOR_MISS) {
        assert(m_config.m_cache_type == SECTOR);
        ((sector_cache_block *)m_lines[idx])->allocate_sector(time, mask);
    }

    // 下面都是tag匹配的代码

    /*之前是dirty并且填充完新的data后不是dirty，才将m_dirty减一 */
    if (before && !m_lines[idx]->is_modified_line()) {
        m_dirty--;
    }

    /*判断新的cacheLine是否为dirty*/
    before = m_lines[idx]->is_modified_line();
    m_lines[idx]->fill(time, mask, byte_mask);
    if (m_lines[idx]->is_modified_line() && !before) {
        m_dirty++;
    }
}

/**
 * 采用ON_MISS策略，之前分配的cacheLine通过参数index传入
 */
void tag_array::fill(unsigned index, unsigned time, mem_fetch *mf) {
    assert(m_config.m_alloc_policy == ON_MISS);
    /*之前选的cacheLine是否为dirty data */
    bool before = m_lines[index]->is_modified_line();
    /* */
    m_lines[index]->fill(time, mf->get_access_sector_mask(),mf->get_access_byte_mask());
    if (m_lines[index]->is_modified_line() && !before) {
        m_dirty++;
    }
}

// TODO: we need write back the flushed data to the upper level
void tag_array::flush() {
    if (!is_used)
        return;

    for (unsigned i = 0; i < m_config.get_num_lines(); i++)
        if (m_lines[i]->is_modified_line()) {
            for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++) {
                m_lines[i]->set_status(INVALID,mem_access_sector_mask_t().set(j));
            }
        }

    m_dirty = 0;
    is_used = false;
}

/*设置所有cache Line状态为 INVALID */
void tag_array::invalidate() {
    if (!is_used)
        return;

    for (unsigned i = 0; i < m_config.get_num_lines(); i++)
        for (unsigned j = 0; j < SECTOR_CHUNCK_SIZE; j++)
            m_lines[i]->set_status(INVALID, mem_access_sector_mask_t().set(j));

    m_dirty = 0;
    is_used = false;
}

float tag_array::windowed_miss_rate() const {
    unsigned n_access = m_access - m_prev_snapshot_access;
    unsigned n_miss = (m_miss + m_sector_miss) - m_prev_snapshot_miss;
    // unsigned n_pending_hit = m_pending_hit - m_prev_snapshot_pending_hit;

    float missrate = 0.0f;
    if (n_access != 0)
        missrate = (float)(n_miss + m_sector_miss) / n_access;
    return missrate;
}

void tag_array::new_window() {
    m_prev_snapshot_access = m_access;
    m_prev_snapshot_miss = m_miss;
    m_prev_snapshot_miss = m_miss + m_sector_miss;
    m_prev_snapshot_pending_hit = m_pending_hit;
}

void tag_array::print(FILE *stream, unsigned &total_access,
                      unsigned &total_misses) const {
    m_config.print(stream);
    fprintf(stream,
            "\t\tAccess = %d, Miss = %d, Sector_Miss = %d, Total_Miss = %d "
            "(%.3g), PendingHit = %d (%.3g)\n",
            m_access, m_miss, m_sector_miss, (m_miss + m_sector_miss),
            (float)(m_miss + m_sector_miss) / m_access, m_pending_hit,
            (float)m_pending_hit / m_access);
    total_misses += (m_miss + m_sector_miss);
    total_access += m_access;
}

void tag_array::get_stats(unsigned &total_access, unsigned &total_misses,
                          unsigned &total_hit_res,
                          unsigned &total_res_fail) const {
    // Update statistics from the tag array
    total_access = m_access;
    total_misses = (m_miss + m_sector_miss);
    total_hit_res = m_pending_hit;
    total_res_fail = m_res_fail;
}

bool was_write_sent(const std::list<cache_event> &events) {
    for (std::list<cache_event>::const_iterator e = events.begin();
         e != events.end(); e++) {
        if ((*e).m_cache_event_type == WRITE_REQUEST_SENT)
            return true;
    }
    return false;
}

bool was_writeback_sent(const std::list<cache_event> &events,
                        cache_event &wb_event) {
    for (std::list<cache_event>::const_iterator e = events.begin();
         e != events.end(); e++) {
        if ((*e).m_cache_event_type == WRITE_BACK_REQUEST_SENT) {
            wb_event = *e;
            return true;
        }
    }
    return false;
}

bool was_read_sent(const std::list<cache_event> &events) {
    for (std::list<cache_event>::const_iterator e = events.begin();
         e != events.end(); e++) {
        if ((*e).m_cache_event_type == READ_REQUEST_SENT)
            return true;
    }
    return false;
}

bool was_writeallocate_sent(const std::list<cache_event> &events) {
    for (std::list<cache_event>::const_iterator e = events.begin();
         e != events.end(); e++) {
        if ((*e).m_cache_event_type == WRITE_ALLOCATE_SENT)
            return true;
    }
    return false;
}

/**
 * 检查给定的地址block_addr对应的内存请求是否已经被合并到mshr entry，即有已经发送给lower memory的pending request
 */
bool mshr_table::probe(new_addr_type block_addr) const {
    table::const_iterator a = m_data.find(block_addr);
    return a != m_data.end();
}

/**
 * 检查mshr是否还有空间记录新的memory request，要求：
 * - 当前的addr对应的mshr entry存在，检查是否该entry的内存请求合并数量已达到最大值
 * - 是否有空闲的MSHR entry分配给新的addr
 */
bool mshr_table::full(new_addr_type block_addr) const {
    table::const_iterator i = m_data.find(block_addr);
    if (i != m_data.end())
        return i->second.m_list.size() >= m_max_merged;
    else
        return m_data.size() >= m_num_entries;
}

/**
 * 
 * 默认MSHR可以处理当前请求，直接向对应的mshr插入请求，要么分配新的entry，要么插入已有的mshr entry
 * 用assert直接检查
 */
void mshr_table::add(new_addr_type block_addr, mem_fetch *mf) {
    m_data[block_addr].m_list.push_back(mf);
    assert(m_data.size() <= m_num_entries);
    assert(m_data[block_addr].m_list.size() <= m_max_merged);
    // indicate that this MSHR entry contains an atomic operation
    if (mf->isatomic()) {
        m_data[block_addr].m_has_atomic = true;
    }
}

/// check is_read_after_write_pending
bool mshr_table::is_read_after_write_pending(new_addr_type block_addr) {
    std::list<mem_fetch *> my_list = m_data[block_addr].m_list;
    bool write_found = false;
    for (std::list<mem_fetch *>::iterator it = my_list.begin();
         it != my_list.end(); ++it) {
        if ((*it)->is_write()) // Pending Write Request
            write_found = true;
        else if (write_found) // Pending Read Request and we found previous
                              // Write
            return true;
    }

    return false;
}

/**
 * 接收一个新的cache fill response，把cache fill response的信息放入m_current_response
 */
void mshr_table::mark_ready(new_addr_type block_addr, bool &has_atomic) {
    assert(!busy());
    table::iterator a = m_data.find(block_addr);
    assert(a != m_data.end());
    m_current_response.push_back(block_addr);
    has_atomic = a->second.m_has_atomic;
    assert(m_current_response.size() <= m_data.size());
}

/**
 * next_access()主要返回请求返回的响应的mem_fetch并且释放对应的mshr entry
 * 每次都返回m_current_response头部的block_addr在mshr中对应的mem_fetch信息，
 * 只有当block_addr对应的mshr entry合并的请求全部返回，才弹出m_current_response头部的block_addr
*/
mem_fetch *mshr_table::next_access() {
    /*首先断言m_current_response非空 */
    assert(access_ready());
    /*取到头部的响应后的的 memory request的addr */
    new_addr_type block_addr = m_current_response.front();

    /*addr对应的mshr entry必须存在*/
    assert(!m_data[block_addr].m_list.empty());

    /*每次调用next_access()即弹出mshr entry处理的merged memory request的一个request*/
    mem_fetch *result = m_data[block_addr].m_list.front();
    m_data[block_addr].m_list.pop_front();

    /*如果该mshr entry处理的merge request已经全部返回，则释放该cache block对应的entry */
    if (m_data[block_addr].m_list.empty()) {
        // release entry
        m_data.erase(block_addr);
        m_current_response.pop_front();
    }
    return result;
}

void mshr_table::display(FILE *fp) const {
    fprintf(fp, "MSHR contents\n");
    for (table::const_iterator e = m_data.begin(); e != m_data.end(); ++e) {
        unsigned block_addr = e->first;
        fprintf(fp, "MSHR: tag=0x%06x, atomic=%d %zu entries : ", block_addr,
                e->second.m_has_atomic, e->second.m_list.size());
        if (!e->second.m_list.empty()) {
            mem_fetch *mf = e->second.m_list.front();
            fprintf(fp, "%p :", mf);
            mf->print(fp);
        } else {
            fprintf(fp, " no memory requests???\n");
        }
    }
}
/***************************************************************** Caches
 * *****************************************************************/
cache_stats::cache_stats() {
    m_cache_port_available_cycles = 0;
    m_cache_data_port_busy_cycles = 0;
    m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear() {
    ///
    /// Zero out all current cache statistics
    ///
    m_stats.clear();
    m_stats_pw.clear();
    m_fail_stats.clear();

    m_cache_port_available_cycles = 0;
    m_cache_data_port_busy_cycles = 0;
    m_cache_fill_port_busy_cycles = 0;
}

void cache_stats::clear_pw() {
    ///
    /// Zero out per-window cache statistics
    ///
    m_stats_pw.clear();
}

void cache_stats::inc_stats(int access_type, int access_outcome,
                            unsigned long long streamID) {
    ///
    /// Increment the stat corresponding to (access_type, access_outcome) by 1.
    ///
    if (!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    if (m_stats.find(streamID) == m_stats.end()) {
        std::vector<std::vector<unsigned long long>> new_val;
        new_val.resize(NUM_MEM_ACCESS_TYPE);
        for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
            new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
        }
        m_stats.insert(std::pair<unsigned long long,
                                 std::vector<std::vector<unsigned long long>>>(
            streamID, new_val));
    }
    m_stats.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_stats_pw(int access_type, int access_outcome,
                               unsigned long long streamID) {
    ///
    /// Increment the corresponding per-window cache stat
    ///
    if (!check_valid(access_type, access_outcome))
        assert(0 && "Unknown cache access type or access outcome");

    if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
        std::vector<std::vector<unsigned long long>> new_val;
        new_val.resize(NUM_MEM_ACCESS_TYPE);
        for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
            new_val[j].resize(NUM_CACHE_REQUEST_STATUS, 0);
        }
        m_stats_pw.insert(
            std::pair<unsigned long long,
                      std::vector<std::vector<unsigned long long>>>(streamID,
                                                                    new_val));
    }
    m_stats_pw.at(streamID)[access_type][access_outcome]++;
}

void cache_stats::inc_fail_stats(int access_type, int fail_outcome,
                                 unsigned long long streamID) {
    if (!check_fail_valid(access_type, fail_outcome))
        assert(0 && "Unknown cache access type or access fail");

    if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
        std::vector<std::vector<unsigned long long>> new_val;
        new_val.resize(NUM_MEM_ACCESS_TYPE);
        for (unsigned j = 0; j < NUM_MEM_ACCESS_TYPE; ++j) {
            new_val[j].resize(NUM_CACHE_RESERVATION_FAIL_STATUS, 0);
        }
        m_fail_stats.insert(
            std::pair<unsigned long long,
                      std::vector<std::vector<unsigned long long>>>(streamID,
                                                                    new_val));
    }
    m_fail_stats.at(streamID)[access_type][fail_outcome]++;
}

enum cache_request_status
cache_stats::select_stats_status(enum cache_request_status probe,
                                 enum cache_request_status access) const {
    ///
    /// This function selects how the cache access outcome should be counted.
    /// HIT_RESERVED is considered as a MISS in the cores, however, it should be
    /// counted as a HIT_RESERVED in the caches.
    ///
    if (probe == HIT_RESERVED && access != RESERVATION_FAIL)
        return probe;
    else if (probe == SECTOR_MISS && access == MISS)
        return probe;
    else
        return access;
}

unsigned long long &cache_stats::operator()(int access_type, int access_outcome,
                                            bool fail_outcome,
                                            unsigned long long streamID) {
    ///
    /// Simple method to read/modify the stat corresponding to (access_type,
    /// access_outcome) Used overloaded () to avoid the need for separate
    /// read/write member functions
    ///
    if (fail_outcome) {
        if (!check_fail_valid(access_type, access_outcome))
            assert(0 && "Unknown cache access type or fail outcome");

        return m_fail_stats.at(streamID)[access_type][access_outcome];
    } else {
        if (!check_valid(access_type, access_outcome))
            assert(0 && "Unknown cache access type or access outcome");

        return m_stats.at(streamID)[access_type][access_outcome];
    }
}

unsigned long long cache_stats::operator()(int access_type, int access_outcome,
                                           bool fail_outcome,
                                           unsigned long long streamID) const {
    ///
    /// Const accessor into m_stats.
    ///
    if (fail_outcome) {
        if (!check_fail_valid(access_type, access_outcome))
            assert(0 && "Unknown cache access type or fail outcome");

        return m_fail_stats.at(streamID)[access_type][access_outcome];
    } else {
        if (!check_valid(access_type, access_outcome))
            assert(0 && "Unknown cache access type or access outcome");

        return m_stats.at(streamID)[access_type][access_outcome];
    }
}

cache_stats cache_stats::operator+(const cache_stats &cs) {
    ///
    /// Overloaded + operator to allow for simple stat accumulation
    ///
    cache_stats ret;
    for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
        unsigned long long streamID = iter->first;
        ret.m_stats.insert(
            std::pair<unsigned long long,
                      std::vector<std::vector<unsigned long long>>>(
                streamID, m_stats.at(streamID)));
    }
    for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
        unsigned long long streamID = iter->first;
        ret.m_stats_pw.insert(
            std::pair<unsigned long long,
                      std::vector<std::vector<unsigned long long>>>(
                streamID, m_stats_pw.at(streamID)));
    }
    for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
        unsigned long long streamID = iter->first;
        ret.m_fail_stats.insert(
            std::pair<unsigned long long,
                      std::vector<std::vector<unsigned long long>>>(
                streamID, m_fail_stats.at(streamID)));
    }
    for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
        unsigned long long streamID = iter->first;
        if (ret.m_stats.find(streamID) == ret.m_stats.end()) {
            ret.m_stats.insert(
                std::pair<unsigned long long,
                          std::vector<std::vector<unsigned long long>>>(
                    streamID, cs.m_stats.at(streamID)));
        } else {
            for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
                for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
                     ++status) {
                    ret.m_stats.at(streamID)[type][status] +=
                        cs(type, status, false, streamID);
                }
            }
        }
    }
    for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end();
         ++iter) {
        unsigned long long streamID = iter->first;
        if (ret.m_stats_pw.find(streamID) == ret.m_stats_pw.end()) {
            ret.m_stats_pw.insert(
                std::pair<unsigned long long,
                          std::vector<std::vector<unsigned long long>>>(
                    streamID, cs.m_stats_pw.at(streamID)));
        } else {
            for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
                for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
                     ++status) {
                    ret.m_stats_pw.at(streamID)[type][status] +=
                        cs(type, status, false, streamID);
                }
            }
        }
    }
    for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
         ++iter) {
        unsigned long long streamID = iter->first;
        if (ret.m_fail_stats.find(streamID) == ret.m_fail_stats.end()) {
            ret.m_fail_stats.insert(
                std::pair<unsigned long long,
                          std::vector<std::vector<unsigned long long>>>(
                    streamID, cs.m_fail_stats.at(streamID)));
        } else {
            for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
                for (unsigned status = 0;
                     status < NUM_CACHE_RESERVATION_FAIL_STATUS; ++status) {
                    ret.m_fail_stats.at(streamID)[type][status] +=
                        cs(type, status, true, streamID);
                }
            }
        }
    }
    ret.m_cache_port_available_cycles =
        m_cache_port_available_cycles + cs.m_cache_port_available_cycles;
    ret.m_cache_data_port_busy_cycles =
        m_cache_data_port_busy_cycles + cs.m_cache_data_port_busy_cycles;
    ret.m_cache_fill_port_busy_cycles =
        m_cache_fill_port_busy_cycles + cs.m_cache_fill_port_busy_cycles;
    return ret;
}

cache_stats &cache_stats::operator+=(const cache_stats &cs) {
    ///
    /// Overloaded += operator to allow for simple stat accumulation
    ///
    for (auto iter = cs.m_stats.begin(); iter != cs.m_stats.end(); ++iter) {
        unsigned long long streamID = iter->first;
        if (m_stats.find(streamID) == m_stats.end()) {
            m_stats.insert(
                std::pair<unsigned long long,
                          std::vector<std::vector<unsigned long long>>>(
                    streamID, cs.m_stats.at(streamID)));
        } else {
            for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
                for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
                     ++status) {
                    m_stats.at(streamID)[type][status] +=
                        cs(type, status, false, streamID);
                }
            }
        }
    }
    for (auto iter = cs.m_stats_pw.begin(); iter != cs.m_stats_pw.end();
         ++iter) {
        unsigned long long streamID = iter->first;
        if (m_stats_pw.find(streamID) == m_stats_pw.end()) {
            m_stats_pw.insert(
                std::pair<unsigned long long,
                          std::vector<std::vector<unsigned long long>>>(
                    streamID, cs.m_stats_pw.at(streamID)));
        } else {
            for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
                for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
                     ++status) {
                    m_stats_pw.at(streamID)[type][status] +=
                        cs(type, status, false, streamID);
                }
            }
        }
    }
    for (auto iter = cs.m_fail_stats.begin(); iter != cs.m_fail_stats.end();
         ++iter) {
        unsigned long long streamID = iter->first;
        if (m_fail_stats.find(streamID) == m_fail_stats.end()) {
            m_fail_stats.insert(
                std::pair<unsigned long long,
                          std::vector<std::vector<unsigned long long>>>(
                    streamID, cs.m_fail_stats.at(streamID)));
        } else {
            for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
                for (unsigned status = 0;
                     status < NUM_CACHE_RESERVATION_FAIL_STATUS; ++status) {
                    m_fail_stats.at(streamID)[type][status] +=
                        cs(type, status, true, streamID);
                }
            }
        }
    }
    m_cache_port_available_cycles += cs.m_cache_port_available_cycles;
    m_cache_data_port_busy_cycles += cs.m_cache_data_port_busy_cycles;
    m_cache_fill_port_busy_cycles += cs.m_cache_fill_port_busy_cycles;
    return *this;
}

void cache_stats::print_stats(FILE *fout, unsigned long long streamID,
                              const char *cache_name) const {
    ///
    /// For a given CUDA stream, print out each non-zero cache statistic for
    /// every memory access type and status "cache_name" defaults to
    /// "Cache_stats" when no argument is provided, otherwise the provided name
    /// is used. The printed format is
    /// "<cache_name>[<request_type>][<request_status>] = <stat_value>"
    /// Specify streamID to be -1 to print every stream.

    std::vector<unsigned> total_access;
    std::string m_cache_name = cache_name;
    for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
        unsigned long long streamid = iter->first;
        // when streamID is specified, skip stats for all other streams,
        // otherwise, print stats from all streams
        if ((streamID != -1) && (streamid != streamID))
            continue;
        total_access.clear();
        total_access.resize(NUM_MEM_ACCESS_TYPE, 0);
        for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
            for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
                 ++status) {
                fprintf(
                    fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                    mem_access_type_str((enum mem_access_type)type),
                    cache_request_status_str((enum cache_request_status)status),
                    m_stats.at(streamid)[type][status]);

                if (status != RESERVATION_FAIL && status != MSHR_HIT)
                    // MSHR_HIT is a special type of SECTOR_MISS
                    // so its already included in the SECTOR_MISS
                    total_access[type] += m_stats.at(streamid)[type][status];
            }
        }
        for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
            if (total_access[type] > 0)
                fprintf(fout, "\t%s[%s][%s] = %u\n", m_cache_name.c_str(),
                        mem_access_type_str((enum mem_access_type)type),
                        "TOTAL_ACCESS", total_access[type]);
        }
    }
}

void cache_stats::print_fail_stats(FILE *fout, unsigned long long streamID,
                                   const char *cache_name) const {
    std::string m_cache_name = cache_name;
    for (auto iter = m_fail_stats.begin(); iter != m_fail_stats.end(); ++iter) {
        unsigned long long streamid = iter->first;
        // when streamID is specified, skip stats for all other streams,
        // otherwise, print stats from all streams
        if ((streamID != -1) && (streamid != streamID))
            continue;
        for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
            for (unsigned fail = 0; fail < NUM_CACHE_RESERVATION_FAIL_STATUS;
                 ++fail) {
                if (m_fail_stats.at(streamid)[type][fail] > 0) {
                    fprintf(fout, "\t%s[%s][%s] = %llu\n", m_cache_name.c_str(),
                            mem_access_type_str((enum mem_access_type)type),
                            cache_fail_status_str(
                                (enum cache_reservation_fail_reason)fail),
                            m_fail_stats.at(streamid)[type][fail]);
                }
            }
        }
    }
}

void cache_sub_stats::print_port_stats(FILE *fout,
                                       const char *cache_name) const {
    float data_port_util = 0.0f;
    if (port_available_cycles > 0) {
        data_port_util = (float)data_port_busy_cycles / port_available_cycles;
    }
    fprintf(fout, "%s_data_port_util = %.3f\n", cache_name, data_port_util);
    float fill_port_util = 0.0f;
    if (port_available_cycles > 0) {
        fill_port_util = (float)fill_port_busy_cycles / port_available_cycles;
    }
    fprintf(fout, "%s_fill_port_util = %.3f\n", cache_name, fill_port_util);
}

unsigned long long
cache_stats::get_stats(enum mem_access_type *access_type,
                       unsigned num_access_type,
                       enum cache_request_status *access_status,
                       unsigned num_access_status) const {
    ///
    /// Returns a sum of the stats corresponding to each "access_type" and
    /// "access_status" pair. "access_type" is an array of "num_access_type"
    /// mem_access_types. "access_status" is an array of "num_access_status"
    /// cache_request_statuses.
    ///
    unsigned long long total = 0;
    for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
        unsigned long long streamID = iter->first;
        for (unsigned type = 0; type < num_access_type; ++type) {
            for (unsigned status = 0; status < num_access_status; ++status) {
                if (!check_valid((int)access_type[type],
                                 (int)access_status[status]))
                    assert(0 && "Unknown cache access type or access outcome");
                total += m_stats.at(
                    streamID)[access_type[type]][access_status[status]];
            }
        }
    }
    return total;
}

void cache_stats::get_sub_stats(struct cache_sub_stats &css) const {
    ///
    /// Overwrites "css" with the appropriate statistics from this cache.
    ///
    struct cache_sub_stats t_css;
    t_css.clear();

    for (auto iter = m_stats.begin(); iter != m_stats.end(); ++iter) {
        unsigned long long streamID = iter->first;
        for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
            for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
                 ++status) {
                if (status == HIT || status == MISS || status == SECTOR_MISS ||
                    status == HIT_RESERVED)
                    t_css.accesses += m_stats.at(streamID)[type][status];

                if (status == MISS || status == SECTOR_MISS)
                    t_css.misses += m_stats.at(streamID)[type][status];

                if (status == HIT_RESERVED)
                    t_css.pending_hits += m_stats.at(streamID)[type][status];

                if (status == RESERVATION_FAIL)
                    t_css.res_fails += m_stats.at(streamID)[type][status];
            }
        }
    }

    t_css.port_available_cycles = m_cache_port_available_cycles;
    t_css.data_port_busy_cycles = m_cache_data_port_busy_cycles;
    t_css.fill_port_busy_cycles = m_cache_fill_port_busy_cycles;

    css = t_css;
}

void cache_stats::get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
    ///
    /// Overwrites "css" with the appropriate statistics from this cache.
    ///
    struct cache_sub_stats_pw t_css;
    t_css.clear();

    for (auto iter = m_stats_pw.begin(); iter != m_stats_pw.end(); ++iter) {
        unsigned long long streamID = iter->first;
        for (unsigned type = 0; type < NUM_MEM_ACCESS_TYPE; ++type) {
            for (unsigned status = 0; status < NUM_CACHE_REQUEST_STATUS;
                 ++status) {
                if (status == HIT || status == MISS || status == SECTOR_MISS ||
                    status == HIT_RESERVED)
                    t_css.accesses += m_stats_pw.at(streamID)[type][status];

                if (status == HIT) {
                    if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
                        type == INST_ACC_R) {
                        t_css.read_hits +=
                            m_stats_pw.at(streamID)[type][status];
                    } else if (type == GLOBAL_ACC_W) {
                        t_css.write_hits +=
                            m_stats_pw.at(streamID)[type][status];
                    }
                }

                if (status == MISS || status == SECTOR_MISS) {
                    if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
                        type == INST_ACC_R) {
                        t_css.read_misses +=
                            m_stats_pw.at(streamID)[type][status];
                    } else if (type == GLOBAL_ACC_W) {
                        t_css.write_misses +=
                            m_stats_pw.at(streamID)[type][status];
                    }
                }

                if (status == HIT_RESERVED) {
                    if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
                        type == INST_ACC_R) {
                        t_css.read_pending_hits +=
                            m_stats_pw.at(streamID)[type][status];
                    } else if (type == GLOBAL_ACC_W) {
                        t_css.write_pending_hits +=
                            m_stats_pw.at(streamID)[type][status];
                    }
                }

                if (status == RESERVATION_FAIL) {
                    if (type == GLOBAL_ACC_R || type == CONST_ACC_R ||
                        type == INST_ACC_R) {
                        t_css.read_res_fails +=
                            m_stats_pw.at(streamID)[type][status];
                    } else if (type == GLOBAL_ACC_W) {
                        t_css.write_res_fails +=
                            m_stats_pw.at(streamID)[type][status];
                    }
                }
            }
        }
    }

    css = t_css;
}

bool cache_stats::check_valid(int type, int status) const {
    ///
    /// Verify a valid access_type/access_status
    ///
    if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (status >= 0) &&
        (status < NUM_CACHE_REQUEST_STATUS))
        return true;
    else
        return false;
}

bool cache_stats::check_fail_valid(int type, int fail) const {
    ///
    /// Verify a valid access_type/access_status
    ///
    if ((type >= 0) && (type < NUM_MEM_ACCESS_TYPE) && (fail >= 0) &&
        (fail < NUM_CACHE_RESERVATION_FAIL_STATUS))
        return true;
    else
        return false;
}

void cache_stats::sample_cache_port_utility(bool data_port_busy,
                                            bool fill_port_busy) {
    m_cache_port_available_cycles += 1;
    if (data_port_busy) {
        m_cache_data_port_busy_cycles += 1;
    }
    if (fill_port_busy) {
        m_cache_fill_port_busy_cycles += 1;
    }
}

baseline_cache::bandwidth_management::bandwidth_management(cache_config &config)
    : m_config(config) {
    m_data_port_occupied_cycles = 0;
    m_fill_port_occupied_cycles = 0;
}

/// use the data port based on the outcome and events generated by the mem_fetch
/// request
void baseline_cache::bandwidth_management::use_data_port(
    mem_fetch *mf, enum cache_request_status outcome,
    const std::list<cache_event> &events) {
    unsigned data_size = mf->get_data_size();
    unsigned port_width = m_config.m_data_port_width;
    switch (outcome) {
    case HIT: {
        unsigned data_cycles =
            data_size / port_width + ((data_size % port_width > 0) ? 1 : 0);
        m_data_port_occupied_cycles += data_cycles;
    } break;
    case HIT_RESERVED:
    case MISS: {
        // the data array is accessed to read out the entire line for write-back
        // in case of sector cache we need to write bank only the modified
        // sectors
        cache_event ev(WRITE_BACK_REQUEST_SENT);
        if (was_writeback_sent(events, ev)) {
            unsigned data_cycles =
                ev.m_evicted_block.m_modified_size / port_width;
            m_data_port_occupied_cycles += data_cycles;
        }
    } break;
    case SECTOR_MISS:
    case RESERVATION_FAIL:
        // Does not consume any port bandwidth
        break;
    default:
        assert(0);
        break;
    }
}

// use the fill port
void baseline_cache::bandwidth_management::use_fill_port(mem_fetch *mf) {
    // assume filling the entire line with the returned request
    unsigned fill_cycles = m_config.get_atom_sz() / m_config.m_data_port_width;
    m_fill_port_occupied_cycles += fill_cycles;
}

// called every cache cycle to free up the ports
void baseline_cache::bandwidth_management::replenish_port_bandwidth() {
    if (m_data_port_occupied_cycles > 0) {
        m_data_port_occupied_cycles -= 1;
    }
    assert(m_data_port_occupied_cycles >= 0);

    if (m_fill_port_occupied_cycles > 0) {
        m_fill_port_occupied_cycles -= 1;
    }
    assert(m_fill_port_occupied_cycles >= 0);
}

// query for data port availability
bool baseline_cache::bandwidth_management::data_port_free() const {
    return (m_data_port_occupied_cycles == 0);
}

// query for fill port availability
bool baseline_cache::bandwidth_management::fill_port_free() const {
    return (m_fill_port_occupied_cycles == 0);
}

/**
 * 运行一个cycle，把本周期能处理cache miss的request发送给下级请求
 * 即把mem_fetch从m_miss_queue移动到m_memport
 * */
void baseline_cache::cycle() {
    /*m_miss_queue不为空，则将队首的请求发送到下一级内存 */
    if (!m_miss_queue.empty()) {
        mem_fetch *mf = m_miss_queue.front();
        if (!m_memport->full(mf->size(), mf->get_is_write())) {
            m_miss_queue.pop_front();
            m_memport->push(mf);
        }
    }
    bool data_port_busy = !m_bandwidth_management.data_port_free();
    bool fill_port_busy = !m_bandwidth_management.fill_port_free();
    m_stats.sample_cache_port_utility(data_port_busy, fill_port_busy);
    m_bandwidth_management.replenish_port_bandwidth();
}

/**
 * fill()处理接收到lower memory level的响应后如何修改cache block status以及释放对应的mshr entry
 * 1. 对于sector cache要等待最后一个mf返回才说明取回整个cache line的数据，而对于line cache，返回的mf肯定对应一个cache line。
 * 2. 调用tag_array->fill()完成填充
 */
void baseline_cache::fill(mem_fetch *mf, unsigned time) {
    /**对于sector cache，需要看当前mf是否是一个大mf分割后返回的最后一个小mf；
     * 对于line cache的话，当前返回mf一定是一整个cache line的数据*/
    if (m_config.m_mshr_type == SECTOR_ASSOC) {
        assert(mf->get_original_mf());
        /*m_extra_mf_fields是hashmap，用来映射mem_fetch-->extra_mf_fields，
        其中extra_mf_fields中的pending_read字段记录拆分的需要接收的小mf数量*/
        extra_mf_fields_lookup::iterator e =
            m_extra_mf_fields.find(mf->get_original_mf());
        assert(e != m_extra_mf_fields.end());
        /*直到最后一个 */
        e->second.pending_read--;
        /*丢弃当前mem_fetch，直到最后一个到来*/
        if (e->second.pending_read > 0) {
            delete mf;
            return;
        } else {
            /*如果m_extra_mf_fields[mf].pending_read等于0，说明这个mf已经是最后一个分割的mf了，
            不需要再等待其他与mf相同请求的数据，可以填充到cache中。 */
            mem_fetch *temp = mf;
            /*得到最初的mem_fetch */
            mf = mf->get_original_mf();
            delete temp;
        }
    }

    /*设置最终填充的mem_fetch属性，并将最后得到的mem_fetch填充到cache中 */
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert(e != m_extra_mf_fields.end());
    assert(e->second.m_valid);
    mf->set_data_size(e->second.m_data_size);
    mf->set_addr(e->second.m_addr);

    /**
     * (1) allocate-on-miss：当发生未完成的cache miss时，需要为未完成的miss分配cache line slot、MSHR和miss队列条目。
     * (2) allocate-on-fill：当发生未完成的cache miss时，需要为未完成的miss分配MSHR和miss队列条目，
     * 但当所需数据从较低内存级别返回时，会选择受害者cache line slot替换
     */
    if (m_config.m_alloc_policy == ON_MISS)
        m_tag_array->fill(e->second.m_cache_index, time, mf);
    else if (m_config.m_alloc_policy == ON_FILL) {
        m_tag_array->fill(e->second.m_block_addr, time, mf, mf->is_write());
    } else
        abort();
    bool has_atomic = false;
    m_mshrs.mark_ready(e->second.m_block_addr, has_atomic);
    if (has_atomic) {
        assert(m_config.m_alloc_policy == ON_MISS);
        cache_block_t *block = m_tag_array->get_block(e->second.m_cache_index);
        if (!block->is_modified_line()) {
            m_tag_array->inc_dirty();
        }
        block->set_status(MODIFIED,
                          mf->get_access_sector_mask()); // mark line as dirty
                                                         // for atomic operation
        block->set_byte_mask(mf);
    }
    m_extra_mf_fields.erase(mf);
    m_bandwidth_management.use_fill_port(mf);
}

// Checks if mf is waiting to be filled by lower memory level
bool baseline_cache::waiting_for_fill(mem_fetch *mf) {
    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    return e != m_extra_mf_fields.end();
}

void baseline_cache::print(FILE *fp, unsigned &accesses,
                           unsigned &misses) const {
    fprintf(fp, "Cache %s:\t", m_name.c_str());
    m_tag_array->print(fp, accesses, misses);
}

void baseline_cache::display_state(FILE *fp) const {
    fprintf(fp, "Cache %s:\n", m_name.c_str());
    m_mshrs.display(fp);
    fprintf(fp, "\n");
}

void baseline_cache::inc_aggregated_stats(cache_request_status status,
                                          cache_request_status cache_status,
                                          mem_fetch *mf,
                                          enum cache_gpu_level level) {
    if (level == L1_GPU_CACHE) {
        m_gpu->aggregated_l1_stats.inc_stats(
            mf->get_streamID(), mf->get_access_type(),
            m_gpu->aggregated_l1_stats.select_stats_status(status,
                                                           cache_status));
    } else if (level == L2_GPU_CACHE) {
        m_gpu->aggregated_l2_stats.inc_stats(
            mf->get_streamID(), mf->get_access_type(),
            m_gpu->aggregated_l2_stats.select_stats_status(status,
                                                           cache_status));
    }
}

void baseline_cache::inc_aggregated_fail_stats(
    cache_request_status status, cache_request_status cache_status,
    mem_fetch *mf, enum cache_gpu_level level) {
    if (level == L1_GPU_CACHE) {
        m_gpu->aggregated_l1_stats.inc_fail_stats(
            mf->get_streamID(), mf->get_access_type(),
            m_gpu->aggregated_l1_stats.select_stats_status(status,
                                                           cache_status));
    } else if (level == L2_GPU_CACHE) {
        m_gpu->aggregated_l2_stats.inc_fail_stats(
            mf->get_streamID(), mf->get_access_type(),
            m_gpu->aggregated_l2_stats.select_stats_status(status,
                                                           cache_status));
    }
}

void baseline_cache::inc_aggregated_stats_pw(cache_request_status status,
                                             cache_request_status cache_status,
                                             mem_fetch *mf,
                                             enum cache_gpu_level level) {
    if (level == L1_GPU_CACHE) {
        m_gpu->aggregated_l1_stats.inc_stats_pw(
            mf->get_streamID(), mf->get_access_type(),
            m_gpu->aggregated_l1_stats.select_stats_status(status,
                                                           cache_status));
    } else if (level == L2_GPU_CACHE) {
        m_gpu->aggregated_l2_stats.inc_stats_pw(
            mf->get_streamID(), mf->get_access_type(),
            m_gpu->aggregated_l2_stats.select_stats_status(status,
                                                           cache_status));
    }
}

// Read miss handler without writeback
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
    bool wb = false;
    evicted_block_info e;
    send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb, e,
                      events, read_only, wa);
}

/**
 * send_read_request()用来处理cache read miss的情形
 * 首先检查该请求是否命中MSHR，如果MSHR不命中，MSHR能够处理该request
 *      - bool &do_miss:记录cache miss的请求是否被处理
 */
void baseline_cache::send_read_request(new_addr_type addr,
                                       new_addr_type block_addr,
                                       unsigned cache_index, mem_fetch *mf,
                                       unsigned time, bool &do_miss, bool &wb,
                                       evicted_block_info &evicted,
                                       std::list<cache_event> &events,
                                       bool read_only, bool wa) {
    /*mshr_addr用于索引该请求的addr对应的mshr entry，其中Sector Cache和Line Cache有不同的算法*/
    new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

    /**
     * 查找是否已经有 mshr_addr 的请求被合并到 MSHR。
     * 需要注意，MSHR 中的条目是以 mshr_addr 为索引的，即来自同一个 line（对于非 Sector Cache）
     * 或者来自同一个 sector（对于 Sector Cache）的请求被合并
     * 因为这种 cache 所请求的最小单位分别是一个 line 或者一个 sector
     */
    bool mshr_hit = m_mshrs.probe(mshr_addr);

    /**
     * 检查mshr是否还有空间记录新的memory request，要求：
     *  - 当前的addr对应的mshr entry存在，检查是否该entry的内存请求合并数量已达到最大值
     *  - 是否有空闲的MSHR entry分配给新的addr
     */
    bool mshr_avail = !m_mshrs.full(mshr_addr);

    /*mshr entry命中并且mshr entry未到达最大可合并的内存请求数 */
    if (mshr_hit && mshr_avail) {
        /*使用access更新LRU状态，注意这里使用block_addr*/
        if (read_only)
            m_tag_array->access(block_addr, time, cache_index, mf);
        else
            m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

        /*把该请求合并到mshr entry中*/
        m_mshrs.add(mshr_addr, mf);
        m_stats.inc_stats(mf->get_access_type(), MSHR_HIT, mf->get_streamID());
        do_miss = true;

    } 

    /*mshr entry未命中，但是有空闲的mshr entry可以分配*/
    else if (!mshr_hit && mshr_avail &&
               (m_miss_queue.size() < m_config.m_miss_queue_size)) {
        /*使用access更新LRU状态*/
        if (read_only)
            m_tag_array->access(block_addr, time, cache_index, mf);
        else
            m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);

        /*添加新的mshr entry记录该request*/
        m_mshrs.add(mshr_addr, mf);

        /*由于mshr未命中，同时需要把该数据的请求发送到下一级内存*/
        m_extra_mf_fields[mf] =
            extra_mf_fields(mshr_addr, mf->get_addr(), cache_index,
                            mf->get_data_size(), m_config);
        mf->set_data_size(m_config.get_atom_sz());
        mf->set_addr(mshr_addr);
        /*即把cache miss的请求放入m_miss_queue以等待发给下一级内存*/
        m_miss_queue.push_back(mf);
        mf->set_status(m_miss_queue_status, time);
        if (!wa)
            events.push_back(cache_event(READ_REQUEST_SENT));

        do_miss = true;
    } 
    /*记录mshr不可用的日志*/
    else if (mshr_hit && !mshr_avail)
        /* MSHR 命中，但 mshr_addr 对应条目的合并数量达到了最大合并数*/
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                               mf->get_streamID());
    else if (!mshr_hit && !mshr_avail)
        /* MSHR 未命中，且 mshr_addr 没有空闲的 MSHR entry 可将 mshr_addr 插入。*/
        m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                               mf->get_streamID());
    else
        assert(0);
}

// Sends write request to lower level memory (write or writeback)
void data_cache::send_write_request(mem_fetch *mf, cache_event request,
                                    unsigned time,
                                    std::list<cache_event> &events) {
    events.push_back(request);
    m_miss_queue.push_back(mf);
    mf->set_status(m_miss_queue_status, time);
}

/**
 * 只有该sector对应的byte的write mask全部为0，即说明该sector的字节对应的write mask全部清除；
 * 对于read request通常为32 byte，要求write mask为空
 * 对于gpgpu-sim 4.0采用的azy fetch-on-read策略，要求：
 *      1.当一个modified sector接受一个sector read request时，首先只有write-mask全部置1，这样才是readable
 *      2.如果write mask不空，则向低级内存发送该sector对应的fetch请求去和write-mask的byte合并，才能标记sector为readable
 */
void data_cache::update_m_readable(mem_fetch *mf, unsigned cache_index) {
    cache_block_t *block = m_tag_array->get_block(cache_index);
    for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
        if (mf->get_access_sector_mask().test(i)) {
            bool all_set = true;
            for (unsigned k = i * SECTOR_SIZE; k < (i + 1) * SECTOR_SIZE; k++) {
                // If any bit in the byte mask (within the sector) is not set,
                // the sector is unreadable
                if (!block->get_dirty_byte_mask().test(k)) {
                    all_set = false;
                    break;
                }
            }

            /*如果所有的byte mask位全都设置为dirty了，则将该sector可设置为可读，因为当前的 sector已经是全部更新为最新值了*/
            if (all_set)
                block->set_m_readable(true, mf->get_access_sector_mask());
        }
    }
}

/****** Write-hit functions (Set by config file) ******/

// Write-back hit: Mark block as modified

/**
 * WRITE_BACK策略：只需要将data写入当前cache不需要写入下一级缓存，并标记cacheLine为modified
 */
cache_request_status data_cache::wr_hit_wb(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
    
    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr, time, cache_index, mf); // update LRU state
    cache_block_t *block = m_tag_array->get_block(cache_index);

    /*如果该block之前不是modified，则增加modified的cacheline的数量 */
    if (!block->is_modified_line()) {
        m_tag_array->inc_dirty();
    }

    /*设置对应的sector/block 状态为modified */
    block->set_status(MODIFIED, mf->get_access_sector_mask());

    /*设置write mask */
    block->set_byte_mask(mf);
    update_m_readable(mf, cache_index);

    return HIT;
}

// Write-through hit: Directly send request to lower level memory
cache_request_status data_cache::wr_hit_wt(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
    if (miss_queue_full(0)) {
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
        return RESERVATION_FAIL; // cannot handle request this cycle
    }

    new_addr_type block_addr = m_config.block_addr(addr);
    m_tag_array->access(block_addr, time, cache_index, mf); // update LRU state
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
        m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    update_m_readable(mf, cache_index);

    // generate a write-through
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

    return HIT;
}

/// Write-evict hit: Send request to lower level memory and invalidate
/// corresponding block
cache_request_status data_cache::wr_hit_we(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status) {
    if (miss_queue_full(0)) {
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
        return RESERVATION_FAIL; // cannot handle request this cycle
    }

    // generate a write-through/evict
    cache_block_t *block = m_tag_array->get_block(cache_index);
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

    // Invalidate block
    block->set_status(INVALID, mf->get_access_sector_mask());

    return HIT;
}

/// Global write-evict, local write-back: Useful for private caches
enum cache_request_status data_cache::wr_hit_global_we_local_wb(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
    bool evict =
        (mf->get_access_type() ==
         GLOBAL_ACC_W); // evict a line that hits on global memory write
    if (evict)
        return wr_hit_we(addr, cache_index, mf, time, events,
                         status); // Write-evict
    else
        return wr_hit_wb(addr, cache_index, mf, time, events,
                         status); // Write-back
}

/****** Write-miss functions (Set by config file) ******/

/// Write-allocate miss: Send write request to lower level memory
// and send a read request for the same block
enum cache_request_status data_cache::wr_miss_wa_naive(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
    new_addr_type block_addr = m_config.block_addr(addr);
    new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

    // Write allocate, maximum 3 requests (write miss, read request, write back
    // request) Conservatively ensure the worst-case request can be handled this
    // cycle
    bool mshr_hit = m_mshrs.probe(mshr_addr);
    bool mshr_avail = !m_mshrs.full(mshr_addr);
    if (miss_queue_full(2) ||
        (!(mshr_hit && mshr_avail) &&
         !(!mshr_hit && mshr_avail &&
           (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
        // check what is the exactly the failure reason
        if (miss_queue_full(2))
            m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                                   mf->get_streamID());
        else if (mshr_hit && !mshr_avail)
            m_stats.inc_fail_stats(mf->get_access_type(), MSHR_MERGE_ENRTY_FAIL,
                                   mf->get_streamID());
        else if (!mshr_hit && !mshr_avail)
            m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                                   mf->get_streamID());
        else
            assert(0);

        return RESERVATION_FAIL;
    }

    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
    // Tries to send write allocate request, returns true on success and false
    // on failure if(!send_write_allocate(mf, addr, block_addr, cache_index,
    // time, events))
    //    return RESERVATION_FAIL;

    const mem_access_t *ma = new mem_access_t(
        m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
        false, // Now performing a read
        mf->get_access_warp_mask(), mf->get_access_byte_mask(),
        mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

    mem_fetch *n_mf = new mem_fetch(
        *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
        mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
        m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle);

    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;

    // Send read request resulting from write miss
    send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss, wb,
                      evicted, events, false, true);

    events.push_back(cache_event(WRITE_ALLOCATE_SENT));

    if (do_miss) {
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
            assert(status == MISS); // SECTOR_MISS and HIT_RESERVED should not
                                    // send write back
            mem_fetch *wb = m_memfetch_creator->alloc(
                evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
                evicted.m_byte_mask, evicted.m_sector_mask,
                evicted.m_modified_size, true,
                m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
                NULL, mf->get_streamID());
            // the evicted block may have wrong chip id when advanced L2 hashing
            // is used, so set the right chip address from the original mf
            wb->set_chip(mf->get_tlx_addr().chip);
            wb->set_partition(mf->get_tlx_addr().sub_partition);
            send_write_request(wb,
                               cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                               time, events);
        }
        return MISS;
    }

    return RESERVATION_FAIL;
}

enum cache_request_status data_cache::wr_miss_wa_fetch_on_write(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
    new_addr_type block_addr = m_config.block_addr(addr);
    new_addr_type mshr_addr = m_config.mshr_addr(mf->get_addr());

    if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
        // if the request writes to the whole cache line/sector, then, write and
        // set cache line Modified. and no need to send read request to memory
        // or reserve mshr

        if (miss_queue_full(0)) {
            m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                                   mf->get_streamID());
            return RESERVATION_FAIL; // cannot handle request this cycle
        }

        bool wb = false;
        evicted_block_info evicted;

        cache_request_status status =
            m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
        assert(status != HIT);
        cache_block_t *block = m_tag_array->get_block(cache_index);
        if (!block->is_modified_line()) {
            m_tag_array->inc_dirty();
        }
        block->set_status(MODIFIED, mf->get_access_sector_mask());
        block->set_byte_mask(mf);
        if (status == HIT_RESERVED)
            block->set_ignore_on_fill(true, mf->get_access_sector_mask());

        if (status != RESERVATION_FAIL) {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
                mem_fetch *wb = m_memfetch_creator->alloc(
                    evicted.m_block_addr, m_wrbk_type,
                    mf->get_access_warp_mask(), evicted.m_byte_mask,
                    evicted.m_sector_mask, evicted.m_modified_size, true,
                    m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
                    NULL, mf->get_streamID());
                // the evicted block may have wrong chip id when advanced L2
                // hashing  is used, so set the right chip address from the
                // original mf
                wb->set_chip(mf->get_tlx_addr().chip);
                wb->set_partition(mf->get_tlx_addr().sub_partition);
                send_write_request(
                    wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted), time,
                    events);
            }
            return MISS;
        }
        return RESERVATION_FAIL;
    } else {
        bool mshr_hit = m_mshrs.probe(mshr_addr);
        bool mshr_avail = !m_mshrs.full(mshr_addr);
        if (miss_queue_full(1) ||
            (!(mshr_hit && mshr_avail) &&
             !(!mshr_hit && mshr_avail &&
               (m_miss_queue.size() < m_config.m_miss_queue_size)))) {
            // check what is the exactly the failure reason
            if (miss_queue_full(1))
                m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                                       mf->get_streamID());
            else if (mshr_hit && !mshr_avail)
                m_stats.inc_fail_stats(mf->get_access_type(),
                                       MSHR_MERGE_ENRTY_FAIL,
                                       mf->get_streamID());
            else if (!mshr_hit && !mshr_avail)
                m_stats.inc_fail_stats(mf->get_access_type(), MSHR_ENRTY_FAIL,
                                       mf->get_streamID());
            else
                assert(0);

            return RESERVATION_FAIL;
        }

        // prevent Write - Read - Write in pending mshr
        // allowing another write will override the value of the first write,
        // and the pending read request will read incorrect result from the
        // second write
        if (m_mshrs.probe(mshr_addr) &&
            m_mshrs.is_read_after_write_pending(mshr_addr) && mf->is_write()) {
            // assert(0);
            m_stats.inc_fail_stats(mf->get_access_type(), MSHR_RW_PENDING,
                                   mf->get_streamID());
            return RESERVATION_FAIL;
        }

        const mem_access_t *ma = new mem_access_t(
            m_wr_alloc_type, mf->get_addr(), m_config.get_atom_sz(),
            false, // Now performing a read
            mf->get_access_warp_mask(), mf->get_access_byte_mask(),
            mf->get_access_sector_mask(), m_gpu->gpgpu_ctx);

        mem_fetch *n_mf = new mem_fetch(
            *ma, NULL, mf->get_streamID(), mf->get_ctrl_size(), mf->get_wid(),
            mf->get_sid(), mf->get_tpc(), mf->get_mem_config(),
            m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, NULL, mf);

        new_addr_type block_addr = m_config.block_addr(addr);
        bool do_miss = false;
        bool wb = false;
        evicted_block_info evicted;
        send_read_request(addr, block_addr, cache_index, n_mf, time, do_miss,
                          wb, evicted, events, false, true);

        cache_block_t *block = m_tag_array->get_block(cache_index);
        block->set_modified_on_fill(true, mf->get_access_sector_mask());
        block->set_byte_mask_on_fill(true);

        events.push_back(cache_event(WRITE_ALLOCATE_SENT));

        if (do_miss) {
            // If evicted block is modified and not a write-through
            // (already modified lower level)
            if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
                mem_fetch *wb = m_memfetch_creator->alloc(
                    evicted.m_block_addr, m_wrbk_type,
                    mf->get_access_warp_mask(), evicted.m_byte_mask,
                    evicted.m_sector_mask, evicted.m_modified_size, true,
                    m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
                    NULL, mf->get_streamID());
                // the evicted block may have wrong chip id when advanced L2
                // hashing  is used, so set the right chip address from the
                // original mf
                wb->set_chip(mf->get_tlx_addr().chip);
                wb->set_partition(mf->get_tlx_addr().sub_partition);
                send_write_request(
                    wb, cache_event(WRITE_BACK_REQUEST_SENT, evicted), time,
                    events);
            }
            return MISS;
        }
        return RESERVATION_FAIL;
    }
}

/**
 * L2 cache采用lazy fetch-on-read的策略来处理write miss，因为论文发现：
 * all the reads received by L2 caches from the coalescer are 32-byte sectored accesses. 
 * Thus, the read access granularity (32 bytes) is different from the write access granularity (one byte).
 */
enum cache_request_status data_cache::wr_miss_wa_lazy_fetch_on_read(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
    

    new_addr_type block_addr = m_config.block_addr(addr);

    // if the request writes to the whole cache line/sector, then, write and set
    // cache line Modified. and no need to send read request to memory or
    // reserve mshr
    if (miss_queue_full(0)) {
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
        return RESERVATION_FAIL; // cannot handle request this cycle
    }

    if (m_config.m_write_policy == WRITE_THROUGH) {
        send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);
    }

    bool wb = false;
    evicted_block_info evicted;

    cache_request_status m_status =
        m_tag_array->access(block_addr, time, cache_index, wb, evicted, mf);
    
    assert(m_status != HIT);
    cache_block_t *block = m_tag_array->get_block(cache_index);
    if (!block->is_modified_line()) {
        m_tag_array->inc_dirty();
    }
    block->set_status(MODIFIED, mf->get_access_sector_mask());
    block->set_byte_mask(mf);
    if (m_status == HIT_RESERVED) {
        block->set_ignore_on_fill(true, mf->get_access_sector_mask());
        block->set_modified_on_fill(true, mf->get_access_sector_mask());
        block->set_byte_mask_on_fill(true);
    }

    if (mf->get_access_byte_mask().count() == m_config.get_atom_sz()) {
        block->set_m_readable(true, mf->get_access_sector_mask());
    } else {
        block->set_m_readable(false, mf->get_access_sector_mask());
        if (m_status == HIT_RESERVED)
            block->set_readable_on_fill(true, mf->get_access_sector_mask());
    }
    update_m_readable(mf, cache_index);

    if (m_status != RESERVATION_FAIL) {
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
            mem_fetch *wb = m_memfetch_creator->alloc(
                evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
                evicted.m_byte_mask, evicted.m_sector_mask,
                evicted.m_modified_size, true,
                m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
                NULL, mf->get_streamID());
            // the evicted block may have wrong chip id when advanced L2 hashing
            // is used, so set the right chip address from the original mf
            wb->set_chip(mf->get_tlx_addr().chip);
            wb->set_partition(mf->get_tlx_addr().sub_partition);
            send_write_request(wb,
                               cache_event(WRITE_BACK_REQUEST_SENT, evicted),
                               time, events);
        }
        return MISS;
    }
    return RESERVATION_FAIL;
}

/// No write-allocate miss: Simply send write request to lower level memory
enum cache_request_status data_cache::wr_miss_no_wa(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
    if (miss_queue_full(0)) {
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
        return RESERVATION_FAIL; // cannot handle request this cycle
    }

    // on miss, generate write through (no write buffering -- too many threads
    // for that)
    send_write_request(mf, cache_event(WRITE_REQUEST_SENT), time, events);

    return MISS;
}

/****** Read hit functions (Set by config file) ******/

// Baseline read hit: Update LRU status of block.
// Special case for atomic instructions -> Mark block as modified
enum cache_request_status
data_cache::rd_hit_base(new_addr_type addr, unsigned cache_index, mem_fetch *mf,
                        unsigned time, std::list<cache_event> &events,
                        enum cache_request_status status) {
    new_addr_type block_addr = m_config.block_addr(addr);
    /*修改LRU状态 */
    m_tag_array->access(block_addr, time, cache_index, mf);
    // Atomics treated as global read/write requests - Perform read, mark line as MODIFIED
    /**
     * 原子操作从全局存储取值，计算，并写回相同地址三项事务在同一原子操作中完成，因此会修改 cache 的状态为 MODIFIED 
     */
    if (mf->isatomic()) {
        assert(mf->get_access_type() == GLOBAL_ACC_R);
        cache_block_t *block = m_tag_array->get_block(cache_index);
        /**
         * 并判断其是否先前已被 MODIFIED，如果先前未被 MODIFIED，此次原子操作做出 MODIFIED，要增加 dirty 数目，
         * 如果先前 block 已经被 MODIFIED，则先前dirty 数目已经增加过了，就不需要再增加了
         */
        if (!block->is_modified_line()) {
            m_tag_array->inc_dirty();
        }

        // 设置 cache block 的状态为 MODIFIED，以避免其他线程在这个 cache block 上的读写操作
        block->set_status(MODIFIED, mf->get_access_sector_mask()); // mark line as MODIFIED

        // 记录该request对write mask的修改
        block->set_byte_mask(mf);
    }
    return HIT;
}

/****** Read miss functions (Set by config file) ******/

// Baseline read miss: Send read request to lower level memory, perform write-back as necessary

/**
 * 
 */
enum cache_request_status data_cache::rd_miss_base(
    new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
    std::list<cache_event> &events, enum cache_request_status status) {
    /**
     * 读 miss 时，就需要将数据请求发送至下一级存储。这里或许需要真实地向下一级存储发
     * 送读请求，也或许由于 mshr 的存在，可以将数据请求合并进去，这样就不需要真实地向 下一级存储发送读请求
     */
    if (miss_queue_full(1)) {
        // cannot handle request this cycle (might need to generate two requests)
        m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                               mf->get_streamID());
        return RESERVATION_FAIL;
    }

    new_addr_type block_addr = m_config.block_addr(addr);
    bool do_miss = false;
    bool wb = false;
    evicted_block_info evicted;

    /*修改mshr table 和 tag table的状态，尝试向下级内存发送请求*/
    send_read_request(addr, block_addr, cache_index, mf, time, do_miss, wb,
                      evicted, events, false, false);

    /*do_miss记录cache miss的请求是否被处理*/
    if (do_miss) {
        // If evicted block is modified and not a write-through
        // (already modified lower level)
        if (wb && (m_config.m_write_policy != WRITE_THROUGH)) {
            mem_fetch *wb = m_memfetch_creator->alloc(
                evicted.m_block_addr, m_wrbk_type, mf->get_access_warp_mask(),
                evicted.m_byte_mask, evicted.m_sector_mask,
                evicted.m_modified_size, true,
                m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, -1, -1, -1,
                NULL, mf->get_streamID());
            // the evicted block may have wrong chip id when advanced L2 hashing
            // is used, so set the right chip address from the original mf
            wb->set_chip(mf->get_tlx_addr().chip);
            wb->set_partition(mf->get_tlx_addr().sub_partition);
            send_write_request(wb, WRITE_BACK_REQUEST_SENT, time, events);
        }
        return MISS;
    }
    return RESERVATION_FAIL;
}

/**
 * access()首先使用probe()函数得到命中的way index或者不命中时需要驱逐的way index(不改变cache line和LRU状态)
 *  - hit：直接使用tag_array->access更新LRU state和对应的line的state
 *  - miss：由于默认采用allocate-on-miss，所以需要分配下面的资源去处理miss:
 *          • A cache line for replacement
            • Entry in mshr_table
            • Entry in m_miss_queue
        能否分配这些资源在send_read_request()函数进行处理，无法分配则返回RESERVATION_FAIL
 * # new_addr_type addr：访问cache的memory地址
 * # mem_fetch *mf：此次memory request的详细信息
 */
enum cache_request_status
read_only_cache::access(new_addr_type addr, mem_fetch *mf, unsigned time,
                        std::list<cache_event> &events) {
    assert(mf->get_data_size() <= m_config.get_atom_sz());
    assert(m_config.m_write_policy == READ_ONLY);
    assert(!mf->get_is_write());

    /*得到访存请求中的addr对应的cache block addr */
    new_addr_type block_addr = m_config.block_addr(addr);

    /*cache_index得到命中对应的way id或者需要驱逐的way id*/
    unsigned cache_index = (unsigned)-1;

    /*使用probe()*/
    enum cache_request_status status =
        m_tag_array->probe(block_addr, cache_index, mf, mf->is_write());
    
    /*cache_status记录access()的结果*/
    enum cache_request_status cache_status = RESERVATION_FAIL;

    if (status == HIT) {
        /*使用m_tag_array->access()实际更新LRU state和对应的line的state*/
        cache_status = m_tag_array->access(block_addr, time, cache_index,
                                           mf);
    } else if (status != RESERVATION_FAIL) {
        /*HIT_RESERVED/SECTOR_MISS/MISS状态 */
        /*miss_queue_full(0)判断本周期能否处理该miss请求*/
        if (!miss_queue_full(0)) {
            bool do_miss = false;
            /*send_read_request()处理cache miss，可能会发送请求到下级内存*/
            send_read_request(addr, block_addr, cache_index, mf, time, do_miss,
                              events, true, false);
            /*do_miss表示该request能否被mshr处理，不能则为false*/
            if (do_miss)
                cache_status = MISS;
            else
                cache_status = RESERVATION_FAIL;
        } 
        /*miss_queue_full()返回true表示该周期不能处理该request*/
        else {
            cache_status = RESERVATION_FAIL;
            m_stats.inc_fail_stats(mf->get_access_type(), MISS_QUEUE_FULL,
                                   mf->get_streamID());
        }
    } else {
        m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                               mf->get_streamID());
    }

    m_stats.inc_stats(mf->get_access_type(),
                      m_stats.select_stats_status(status, cache_status),
                      mf->get_streamID());
    m_stats.inc_stats_pw(mf->get_access_type(),
                         m_stats.select_stats_status(status, cache_status),
                         mf->get_streamID());
    return cache_status;
}

//! A general function that takes the result of a tag_array probe
//  and performs the correspding functions based on the cache configuration
//  The access fucntion calls this function
enum cache_request_status
data_cache::process_tag_probe(bool wr, enum cache_request_status probe_status,
                              new_addr_type addr, unsigned cache_index,
                              mem_fetch *mf, unsigned time,
                              std::list<cache_event> &events) {
    // Each function pointer ( m_[rd/wr]_[hit/miss] ) is set in the
    // data_cache constructor to reflect the corresponding cache configuration
    // options. Function pointers were used to avoid many long conditional
    // branches resulting from many cache configuration options.
    cache_request_status access_status = probe_status;
    if (wr) { // Write
        if (probe_status == HIT) {
            access_status = (this->*m_wr_hit)(addr, cache_index, mf, time,
                                              events, probe_status);
        } else if ((probe_status != RESERVATION_FAIL) ||
                   (probe_status == RESERVATION_FAIL &&
                    m_config.m_write_alloc_policy == NO_WRITE_ALLOCATE)) {
            access_status = (this->*m_wr_miss)(addr, cache_index, mf, time,
                                               events, probe_status);
        } else {
            // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e
            // all lines are reserved)
            m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                                   mf->get_streamID());
        }
    } else { // Read
        if (probe_status == HIT) {
            access_status = (this->*m_rd_hit)(addr, cache_index, mf, time,
                                              events, probe_status);
        } else if (probe_status != RESERVATION_FAIL) {
            access_status = (this->*m_rd_miss)(addr, cache_index, mf, time,
                                               events, probe_status);
        } else {
            // the only reason for reservation fail here is LINE_ALLOC_FAIL (i.e
            // all lines are reserved)
            m_stats.inc_fail_stats(mf->get_access_type(), LINE_ALLOC_FAIL,
                                   mf->get_streamID());
        }
    }

    m_bandwidth_management.use_data_port(mf, access_status, events);
    return access_status;
}

// Both the L1 and L2 currently use the same access function.
// Differentiation between the two caches is done through configuration
// of caching policies.
// Both the L1 and L2 override this function to provide a means of
// performing actions specific to each cache when such actions are implemnted.
enum cache_request_status data_cache::access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events) {
    assert(mf->get_data_size() <= m_config.get_atom_sz());
    bool wr = mf->get_is_write();
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status probe_status =
        m_tag_array->probe(block_addr, cache_index, mf, mf->is_write(), true);
    enum cache_request_status access_status = process_tag_probe(
        wr, probe_status, addr, cache_index, mf, time, events);
    m_stats.inc_stats(mf->get_access_type(),
                      m_stats.select_stats_status(probe_status, access_status),
                      mf->get_streamID());
    m_stats.inc_stats_pw(
        mf->get_access_type(),
        m_stats.select_stats_status(probe_status, access_status),
        mf->get_streamID());
    return access_status;
}

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at the
/// granularity of individual blocks (Set by GPGPU-Sim configuration file)
/// (the policy used in fermi according to the CUDA manual)
enum cache_request_status l1_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
    return data_cache::access(addr, mf, time, events);
}

// The l2 cache access function calls the base data_cache access
// implementation.  When the L2 needs to diverge from L1, L2 specific
// changes should be made here.
enum cache_request_status l2_cache::access(new_addr_type addr, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events) {
    return data_cache::access(addr, mf, time, events);
}

/// Access function for tex_cache
/// return values: RESERVATION_FAIL if request could not be accepted
/// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
/// since unlike a normal CPU cache, a "HIT" in texture cache does not
/// mean the data is ready (still need to get through fragment fifo)
enum cache_request_status tex_cache::access(new_addr_type addr, mem_fetch *mf,
                                            unsigned time,
                                            std::list<cache_event> &events) {
    if (m_fragment_fifo.full() || m_request_fifo.full() || m_rob.full())
        return RESERVATION_FAIL;

    assert(mf->get_data_size() <= m_config.get_line_sz());

    // at this point, we will accept the request : access tags and immediately
    // allocate line
    new_addr_type block_addr = m_config.block_addr(addr);
    unsigned cache_index = (unsigned)-1;
    enum cache_request_status status =
        m_tags.access(block_addr, time, cache_index, mf);
    enum cache_request_status cache_status = RESERVATION_FAIL;
    assert(status != RESERVATION_FAIL);
    assert(status != HIT_RESERVED); // as far as tags are concerned: HIT or MISS
    m_fragment_fifo.push(
        fragment_entry(mf, cache_index, status == MISS, mf->get_data_size()));
    if (status == MISS) {
        // we need to send a memory request...
        unsigned rob_index = m_rob.push(rob_entry(cache_index, mf, block_addr));
        m_extra_mf_fields[mf] = extra_mf_fields(rob_index, m_config);
        mf->set_data_size(m_config.get_line_sz());
        m_tags.fill(cache_index, time, mf); // mark block as valid
        m_request_fifo.push(mf);
        mf->set_status(m_request_queue_status, time);
        events.push_back(cache_event(READ_REQUEST_SENT));
        cache_status = MISS;
    } else {
        // the value *will* *be* in the cache already
        cache_status = HIT_RESERVED;
    }
    m_stats.inc_stats(mf->get_access_type(),
                      m_stats.select_stats_status(status, cache_status),
                      mf->get_streamID());
    m_stats.inc_stats_pw(mf->get_access_type(),
                         m_stats.select_stats_status(status, cache_status),
                         mf->get_streamID());
    return cache_status;
}

void tex_cache::cycle() {
    // send next request to lower level of memory
    if (!m_request_fifo.empty()) {
        mem_fetch *mf = m_request_fifo.peek();
        if (!m_memport->full(mf->get_ctrl_size(), false)) {
            m_request_fifo.pop();
            m_memport->push(mf);
        }
    }
    // read ready lines from cache
    if (!m_fragment_fifo.empty() && !m_result_fifo.full()) {
        const fragment_entry &e = m_fragment_fifo.peek();
        if (e.m_miss) {
            // check head of reorder buffer to see if data is back from memory
            unsigned rob_index = m_rob.next_pop_index();
            const rob_entry &r = m_rob.peek(rob_index);
            assert(r.m_request == e.m_request);
            // assert( r.m_block_addr ==
            // m_config.block_addr(e.m_request->get_addr())
            // );
            if (r.m_ready) {
                assert(r.m_index == e.m_cache_index);
                m_cache[r.m_index].m_valid = true;
                m_cache[r.m_index].m_block_addr = r.m_block_addr;
                m_result_fifo.push(e.m_request);
                m_rob.pop();
                m_fragment_fifo.pop();
            }
        } else {
            // hit:
            assert(m_cache[e.m_cache_index].m_valid);
            assert(m_cache[e.m_cache_index].m_block_addr ==
                   m_config.block_addr(e.m_request->get_addr()));
            m_result_fifo.push(e.m_request);
            m_fragment_fifo.pop();
        }
    }
}

/// Place returning cache block into reorder buffer
void tex_cache::fill(mem_fetch *mf, unsigned time) {
    if (m_config.m_mshr_type == SECTOR_TEX_FIFO) {
        assert(mf->get_original_mf());
        extra_mf_fields_lookup::iterator e =
            m_extra_mf_fields.find(mf->get_original_mf());
        assert(e != m_extra_mf_fields.end());
        e->second.pending_read--;

        if (e->second.pending_read > 0) {
            // wait for the other requests to come back
            delete mf;
            return;
        } else {
            mem_fetch *temp = mf;
            mf = mf->get_original_mf();
            delete temp;
        }
    }

    extra_mf_fields_lookup::iterator e = m_extra_mf_fields.find(mf);
    assert(e != m_extra_mf_fields.end());
    assert(e->second.m_valid);
    assert(!m_rob.empty());
    mf->set_status(m_rob_status, time);

    unsigned rob_index = e->second.m_rob_index;
    rob_entry &r = m_rob.peek(rob_index);
    assert(!r.m_ready);
    r.m_ready = true;
    r.m_time = time;
    assert(r.m_block_addr == m_config.block_addr(mf->get_addr()));
}

void tex_cache::display_state(FILE *fp) const {
    fprintf(fp, "%s (texture cache) state:\n", m_name.c_str());
    fprintf(fp, "fragment fifo entries  = %u / %u\n", m_fragment_fifo.size(),
            m_fragment_fifo.capacity());
    fprintf(fp, "reorder buffer entries = %u / %u\n", m_rob.size(),
            m_rob.capacity());
    fprintf(fp, "request fifo entries   = %u / %u\n", m_request_fifo.size(),
            m_request_fifo.capacity());
    if (!m_rob.empty())
        fprintf(fp, "reorder buffer contents:\n");
    for (int n = m_rob.size() - 1; n >= 0; n--) {
        unsigned index = (m_rob.next_pop_index() + n) % m_rob.capacity();
        const rob_entry &r = m_rob.peek(index);
        fprintf(fp, "tex rob[%3d] : %s ", index,
                (r.m_ready ? "ready  " : "pending"));
        if (r.m_ready)
            fprintf(fp, "@%6u", r.m_time);
        else
            fprintf(fp, "       ");
        fprintf(fp, "[idx=%4u]", r.m_index);
        r.m_request->print(fp, false);
    }
    if (!m_fragment_fifo.empty()) {
        fprintf(fp, "fragment fifo (oldest) :");
        fragment_entry &f = m_fragment_fifo.peek();
        fprintf(fp, "%s:          ", f.m_miss ? "miss" : "hit ");
        f.m_request->print(fp, false);
    }
}
/******************************************************************************************************************************************/
