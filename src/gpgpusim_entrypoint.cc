// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ivan Sham,
// Andrew Turner, Ali Bakhoda, The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#include "gpgpusim_entrypoint.h"
#include <stdio.h>

#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/cuda-sim.h"
#include "cuda-sim/ptx_ir.h"
#include "cuda-sim/ptx_parser.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"
#include "stream_manager.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static int sg_argc = 3;
static const char *sg_argv[] = {"", "-config", "gpgpusim.config"};

/**
 * gpgpu时序模拟的thread，每次至多运行一个kernel，用于opencl的API调用设置g_sim_signal_start触发thread启动
 */
void *gpgpu_sim_thread_sequential(void *ctx_ptr) {
    gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
    bool done;
    do {
        /*g_sim_signal_start 在哪里被设置 ？*/
        sem_wait(&(ctx->the_gpgpusim->g_sim_signal_start));
        done = true;
        /*当前kernel还有多余的CTA以及当前gpgpu可以运行更多的CTA*/
        if (ctx->the_gpgpusim->g_the_gpu->get_more_cta_left()) {
            done = false;
            /*每执行一个gpgpu的时序模拟，都要格式化化gpgpu_sim的变量*/
            ctx->the_gpgpusim->g_the_gpu->init();
            /*g_the_gpu不断的运行cycle()函数运行一个周期 */
            while (ctx->the_gpgpusim->g_the_gpu->active()) {
                ctx->the_gpgpusim->g_the_gpu->cycle();
                ctx->the_gpgpusim->g_the_gpu->deadlock_check();
            }
            /*执行完一个kernel后打印输出的统计数据 */
            ctx->the_gpgpusim->g_the_gpu->print_stats(
                ctx->the_gpgpusim->g_the_gpu->last_streamID);
            ctx->the_gpgpusim->g_the_gpu->update_stats();
            ctx->print_simulation_time();
        }
        sem_post(&(ctx->the_gpgpusim->g_sim_signal_finish));
    } while (!done);
    sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
    return NULL;
}

static void termination_callback() {
    printf("GPGPU-Sim: *** exit detected ***\n");
    fflush(stdout);
}

/**
 * 对于cuda引用，运行gpgpu_sim_thread_concurrent进行时序模拟
 * 使用active
 */
void *gpgpu_sim_thread_concurrent(void *ctx_ptr) {
    gpgpu_context *ctx = (gpgpu_context *)ctx_ptr;
    atexit(termination_callback);
    // concurrent kernel execution simulation thread
    /**
     * g_sim_done在start_sim_thread()中设置为false，然后陷入while死循环
     * 在
     */
    do {
        if (g_debug_execution >= 3) {
            printf("GPGPU-Sim: *** simulation thread starting and spinning "
                   "waiting for "
                   "work ***\n");
            fflush(stdout);
        }
        while (ctx->the_gpgpusim->g_stream_manager->empty_protected() &&
               !ctx->the_gpgpusim->g_sim_done)
            ;
        if (g_debug_execution >= 3) {
            printf(
                "GPGPU-Sim: ** START simulation thread (detected work) **\n");
            ctx->the_gpgpusim->g_stream_manager->print(stdout);
            fflush(stdout);
        }
        pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
        ctx->the_gpgpusim->g_sim_active = true;
        pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
        bool active = false;
        bool sim_cycles = false;
        /*统计数据初始化, 开始新的时序模拟 */
        ctx->the_gpgpusim->g_the_gpu->init();
        do {
            // check if a kernel has completed
            // launch operation on device if one is pending and can be run

            // Need to break this loop when a kernel completes. This was a
            // source of non-deterministic behaviour in GPGPU-Sim (bug 147).
            // If another stream operation is available, g_the_gpu remains
            // active, causing this loop to not break. If the next operation
            // happens to be another kernel, the gpu is not re-initialized and
            // the inter-kernel behaviour may be incorrect. Check that a kernel
            // has finished and no other kernel is currently running.
            
            /*这里是bug修复相关代码，即在kernel执行完毕后要退出内部循环以重新初始化gpgpu-sim模拟器 */
            /**
             * 这里的g_stream_manager->operation(&sim_cycles)会尝试弹出一个operation去执行
             * 在operation()函数中会根据operation的kernel信息做好进行function or performance sim的准备
             */
            if (ctx->the_gpgpusim->g_stream_manager->operation(&sim_cycles) &&
                !ctx->the_gpgpusim->g_the_gpu->active())
                break;

            // functional simulation
            if (ctx->the_gpgpusim->g_the_gpu->is_functional_sim()) {
                kernel_info_t *kernel = ctx->the_gpgpusim->g_the_gpu->get_functional_kernel();
                assert(kernel);
                /*functional simulation的主体代码，对kernel进行function sim*/
                ctx->the_gpgpusim->gpgpu_ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(*kernel);
                /*结束function sim*/
                ctx->the_gpgpusim->g_the_gpu->finish_functional_sim(kernel);
            }

            // performance simulation
            if (ctx->the_gpgpusim->g_the_gpu->active()) {
                ctx->the_gpgpusim->g_the_gpu->cycle();
                sim_cycles = true;
                ctx->the_gpgpusim->g_the_gpu->deadlock_check();
            } else {
                if (ctx->the_gpgpusim->g_the_gpu->cycle_insn_cta_max_hit()) {
                    ctx->the_gpgpusim->g_stream_manager->stop_all_running_kernels();
                    ctx->the_gpgpusim->g_sim_done = true;
                    ctx->the_gpgpusim->break_limit = true;
                }
            }
            
            /*active = gpu-active() or stream_manager->empty()*/
            active = ctx->the_gpgpusim->g_the_gpu->active() ||
                     !(ctx->the_gpgpusim->g_stream_manager->empty_protected());

        } while (active && !ctx->the_gpgpusim->g_sim_done);

        /** */
        if (g_debug_execution >= 3) {
            printf("GPGPU-Sim: ** STOP simulation thread (no work) **\n");
            fflush(stdout);
        }
        if (sim_cycles) {
            ctx->the_gpgpusim->g_the_gpu->print_stats(
                ctx->the_gpgpusim->g_the_gpu->last_streamID);
            ctx->the_gpgpusim->g_the_gpu->update_stats();
            ctx->print_simulation_time();
        }
        pthread_mutex_lock(&(ctx->the_gpgpusim->g_sim_lock));
        ctx->the_gpgpusim->g_sim_active = false;
        pthread_mutex_unlock(&(ctx->the_gpgpusim->g_sim_lock));
    } while (!ctx->the_gpgpusim->g_sim_done);

    /**时序模拟结束 */
    printf("GPGPU-Sim: *** simulation thread exiting ***\n");
    fflush(stdout);

    if (ctx->the_gpgpusim->break_limit) {
        printf("GPGPU-Sim: ** break due to reaching the maximum cycles (or "
               "instructions) **\n");
        exit(1);
    }

    sem_post(&(ctx->the_gpgpusim->g_sim_signal_exit));
    return NULL;
}

void gpgpu_context::synchronize() {
    printf("GPGPU-Sim: synchronize waiting for inactive GPU simulation\n");
    the_gpgpusim->g_stream_manager->print(stdout);
    fflush(stdout);
    //    sem_wait(&g_sim_signal_finish);
    bool done = false;
    do {
        pthread_mutex_lock(&(the_gpgpusim->g_sim_lock));
        done = (the_gpgpusim->g_stream_manager->empty() &&
                !the_gpgpusim->g_sim_active) ||
               the_gpgpusim->g_sim_done;
        pthread_mutex_unlock(&(the_gpgpusim->g_sim_lock));
    } while (!done);
    printf("GPGPU-Sim: detected inactive GPU simulation thread\n");
    fflush(stdout);
    //    sem_post(&g_sim_signal_start);
}

void gpgpu_context::exit_simulation() {
    the_gpgpusim->g_sim_done = true;
    printf("GPGPU-Sim: exit_simulation called\n");
    fflush(stdout);
    sem_wait(&(the_gpgpusim->g_sim_signal_exit));
    printf("GPGPU-Sim: simulation thread signaled exit\n");
    fflush(stdout);
}

/**
 * gpgpu_ptx_sim_init_perf()主要负责整个GPGPU-Sim模拟器的初始化
 * 任何一个实现在cuda_runtime.cc中的CUDA API的调用都会初始化模拟器，功能如下：
 *   - 读取environment比如debug info, sim mode等
 *   - 解析配置文件
 *   - 初始化 GPU uArch Model，Stream Manager等, 用来初始化全局唯一的
 *      g_the_gpu、g_stream_manager等
*/
gpgpu_sim *gpgpu_context::gpgpu_ptx_sim_init_perf() {
    srand(1);
    /*打印gpgpusim版本信息*/
    print_splash();
    /*解析环境变量 */
    func_sim->read_sim_environment_variables();
    ptx_parser->read_parser_environment_variables();
    option_parser_t opp = option_parser_create();

    ptx_reg_options(opp);
    func_sim->ptx_opcocde_latency_options(opp);

    icnt_reg_options(opp);
    the_gpgpusim->g_the_gpu_config = new gpgpu_sim_config(this);
    the_gpgpusim->g_the_gpu_config->reg_options(
        opp); // register GPU microrachitecture options

    option_parser_cmdline(opp, sg_argc, sg_argv); // parse configuration options
    fprintf(stdout, "GPGPU-Sim: Configuration options:\n\n");
    option_parser_print(opp, stdout);
    // Set the Numeric locale to a standard locale where a decimal point is a
    // "dot" not a "comma" so it does the parsing correctly independent of the
    // system environment variables
    assert(setlocale(LC_NUMERIC, "C"));
    the_gpgpusim->g_the_gpu_config->init();

    /*初始化并得到全局唯一的performance simulation的实例g_the_gpu */
    the_gpgpusim->g_the_gpu =
        new exec_gpgpu_sim(*(the_gpgpusim->g_the_gpu_config), this);
    /*得到全局唯一的cuda stream manager*/
    the_gpgpusim->g_stream_manager = new stream_manager(
        (the_gpgpusim->g_the_gpu), func_sim->g_cuda_launch_blocking);

    the_gpgpusim->g_simulation_starttime = time((time_t *)NULL);

    /*设置各种在后台运行performance simulation的thread需要的信号量为0 */
    sem_init(&(the_gpgpusim->g_sim_signal_start), 0, 0);
    sem_init(&(the_gpgpusim->g_sim_signal_finish), 0, 0);
    sem_init(&(the_gpgpusim->g_sim_signal_exit), 0, 0);

    return the_gpgpusim->g_the_gpu;
}

/**
 * 启动gpu thread用来执行gpgpu-sim仿真
 */
void gpgpu_context::start_sim_thread(int api) {
    if (the_gpgpusim->g_sim_done) {
        /*设置g_sim_done为false */
        the_gpgpusim->g_sim_done = false;
        if (api == 1) {
            pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                           gpgpu_sim_thread_concurrent, (void *)this);
        } else {
            pthread_create(&(the_gpgpusim->g_simulation_thread), NULL,
                           gpgpu_sim_thread_sequential, (void *)this);
        }
    }
}

void gpgpu_context::print_simulation_time() {
    time_t current_time, difference, d, h, m, s;
    current_time = time((time_t *)NULL);
    difference = MAX(current_time - the_gpgpusim->g_simulation_starttime, 1);

    d = difference / (3600 * 24);
    h = difference / 3600 - 24 * d;
    m = difference / 60 - 60 * (h + 24 * d);
    s = difference - 60 * (m + 60 * (h + 24 * d));

    fflush(stderr);
    printf("\n\ngpgpu_simulation_time = %u days, %u hrs, %u min, %u sec (%u "
           "sec)\n",
           (unsigned)d, (unsigned)h, (unsigned)m, (unsigned)s,
           (unsigned)difference);
    printf("gpgpu_simulation_rate = %u (inst/sec)\n",
           (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_insn / difference));
    const unsigned cycles_per_sec =
        (unsigned)(the_gpgpusim->g_the_gpu->gpu_tot_sim_cycle / difference);
    printf("gpgpu_simulation_rate = %u (cycle/sec)\n", cycles_per_sec);
    printf("gpgpu_silicon_slowdown = %ux\n",
           the_gpgpusim->g_the_gpu->shader_clock() * 1000 / cycles_per_sec);
    fflush(stdout);
}

/**
 * 设置g_sim_signal_start信号开始gpgpu-sim的时序模拟
 */
int gpgpu_context::gpgpu_opencl_ptx_sim_main_perf(kernel_info_t *grid) {
    the_gpgpusim->g_the_gpu->launch(grid);
    sem_post(&(the_gpgpusim->g_sim_signal_start));
    sem_wait(&(the_gpgpusim->g_sim_signal_finish));
    return 0;
}

//! Functional simulation of OpenCL
/*!
 * This function call the CUDA PTX functional simulator
 */
int cuda_sim::gpgpu_opencl_ptx_sim_main_func(kernel_info_t *grid) {
    // calling the CUDA PTX simulator, sending the kernel by reference and a
    // flag set to true, the flag used by the function to distinguish OpenCL
    // calls from the CUDA simulation calls which it is needed by the called
    // function to not register the exit the exit of OpenCL kernel as it doesn't
    // register entering in the first place as the CUDA kernels does
    gpgpu_cuda_ptx_sim_main_func(*grid, true);
    return 0;
}
