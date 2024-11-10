#ifndef __gpgpu_context_h__
#define __gpgpu_context_h__
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/cuda_device_runtime.h"
#include "../src/cuda-sim/ptx-stats.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_parser.h"
#include "../src/gpgpusim_entrypoint.h"
#include "cuda_api_object.h"

/**
 * 整个GPGPU-sim模拟过程中最顶层的抽象
 */
class gpgpu_context {
 public:
  gpgpu_context() {
    g_global_allfiles_symbol_table = NULL;
    sm_next_access_uid = 0;
    warp_inst_sm_next_uid = 0;
    operand_info_sm_next_uid = 1;
    kernel_info_m_next_uid = 1;
    g_num_ptx_inst_uid = 0;
    g_ptx_cta_info_uid = 1;
    symbol_sm_next_uid = 1;
    function_info_sm_next_uid = 1;
    debug_tensorcore = 0;
    /*初始化了以下几个class的instance */
    api = new cuda_runtime_api(this);
    ptxinfo = new ptxinfo_data(this);
    ptx_parser = new ptx_recognizer(this);
    the_gpgpusim = new GPGPUsim_ctx(this);
    func_sim = new cuda_sim(this);
    device_runtime = new cuda_device_runtime(this);
    stats = new ptx_stats(this);
  }
  // global list
  symbol_table *g_global_allfiles_symbol_table;
  const char *g_filename;
  unsigned sm_next_access_uid;
  unsigned warp_inst_sm_next_uid;
  unsigned operand_info_sm_next_uid;  // uid for operand_info
  unsigned kernel_info_m_next_uid;    // uid for kernel_info_t
  unsigned g_num_ptx_inst_uid;        // uid for ptx inst inside ptx_instruction
  unsigned long long g_ptx_cta_info_uid;
  unsigned symbol_sm_next_uid;  // uid for symbol
  unsigned function_info_sm_next_uid;

  /*a direct mapping from PC to instruction，直接通过pc获得ptx_instruction*/
  std::vector<ptx_instruction *> s_g_pc_to_insn; 
  bool debug_tensorcore;

  // objects pointers for each file
  cuda_runtime_api *api;
  ptxinfo_data *ptxinfo;
  ptx_recognizer *ptx_parser;

  /*GPGPUsim_ctx是gpgpu_sim的进一步封装, 包括控制信号和stream manager*/
  GPGPUsim_ctx *the_gpgpusim;
  cuda_sim *func_sim;
  cuda_device_runtime *device_runtime;
  ptx_stats *stats;
  // member function list
  void synchronize();
  void exit_simulation();
  void print_simulation_time();
  /*opencl程序通过这个启动时序线程的模拟，每次最多在gpgpusim运行一个kernel*/
  int gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid);
  void cuobjdumpParseBinary(unsigned int handle);
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_string(const char *p,
                                                         unsigned source_num);
  /*在ptx_loader.cc文件中实现，主要通过调用init_parser(const char *)完成parser的初始化*/
  class symbol_table *gpgpu_ptx_sim_load_ptx_from_filename(
      const char *filename);
  void gpgpu_ptx_info_load_from_filename(const char *filename,
                                         unsigned sm_version);
  void gpgpu_ptxinfo_load_from_string(const char *p_for_info,
                                      unsigned source_num,
                                      unsigned sm_version = 20,
                                      int no_of_ptx = 0);
  void print_ptx_file(const char *p, unsigned source_num, const char *filename);
  /*在ptx_parser.cc文件中 symbol_table *gpgpu_context::init_parser()中实现 */
  class symbol_table *init_parser(const char *);
  /*读取各种配置文件，负责整个GPGPU-Sim模拟器的初始化，实例化各种全局变量 */
  class gpgpu_sim *gpgpu_ptx_sim_init_perf();
  /*启动新的thread用于时序模拟 */
  void start_sim_thread(int api);
  /*任何first CUDA/OpenCL API call都会调用GPGPUSim_Init()对整个gpgpusim进行初始化 */
  struct _cuda_device_id *GPGPUSim_Init();
  void ptx_reg_options(option_parser_t opp);

  /*被下面的ptx_fetch_inst(address_type pc)调用*/
  const ptx_instruction *pc_to_instruction(unsigned pc);

  /*从function模拟器中根据pc获取warp_inst_t对象，实现在cuda-sim.cc中 */
  const warp_inst_t *ptx_fetch_inst(address_type pc);
  unsigned translate_pc_to_ptxlineno(unsigned pc);
};
gpgpu_context *GPGPU_Context();

#endif /* __gpgpu_context_h__ */
