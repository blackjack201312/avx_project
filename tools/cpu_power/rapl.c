/*
Copyright (c) 2012, Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Written by Martin Dimitrov, Carl Strickland */



/*! \file rapl.c
 * Intel Power Governor library implementation
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sched.h>
#include <unistd.h>

#include "cpuid.h"
#include "msr.h"
#include "rapl.h"

/* rapl msr availablility */
#define MSR_SUPPORT_MASK 0xff
unsigned char *msr_support_table;

/* Global Variables */
double RAPL_TIME_UNIT;
double RAPL_ENERGY_UNIT;
double RAPL_POWER_UNIT;

unsigned int  num_nodes = 0;
unsigned int  os_cpu_count = 0;
APIC_ID_t *os_map;

/* Pre-computed variables used for time-window calculation */
const double LN2 = 0.69314718055994530941723212145817656807550013436025;
const double A_F[4] = { 1.0, 1.1, 1.2, 1.3 };
const double A_LNF[4] = {
    0.0000000000000000000000000000000000000000000000000000000,
    0.0953101798043249348602046211453853175044059753417968750,
    0.1823215567939545922460098381634452380239963531494140625,
    0.2623642644674910595625760834082029759883880615234375000
};

typedef struct rapl_unit_multiplier_t {
    double power;
    double energy;
    double time;
} rapl_unit_multiplier_t;

typedef struct rapl_power_limit_control_t {
    double       power_limit_watts;
    double       limit_time_window_seconds;
    unsigned int limit_enabled;
    unsigned int clamp_enabled;
    unsigned int lock_enabled;
} rapl_power_limit_control_t;

typedef struct rapl_parameters_t {
    double thermal_spec_power_watts;
    double minimum_power_watts;
    double maximum_power_watts;
    double maximum_limit_time_window_seconds;
} rapl_parameters_t;

// Use numactl in order to find out how many nodes are in the system
// and assign a cpu per node. The reason is that we
//
// Ideally we could include <numa.h> and use the library calls.
// However, I found that numactl-devel is not included by default
// in SLES11.1, which would make it harder to setup the tool.
// This is uglier, but hopefully everyone has numacta
unsigned int  get_num_rapl_nodes_pkg();
unsigned int  get_num_rapl_nodes_pkg();
unsigned int  get_num_rapl_nodes_pkg();

// OS specific
int
bind_context(unsigned int cpu){

    int err =0;
    int ret =0;
    cpu_set_t current_cpu;
    CPU_ZERO(&current_cpu);
    CPU_SET(cpu, &current_cpu);
    err = sched_setaffinity(0, sizeof(cpu_set_t), &current_cpu);
    if(0 != err)
        ret = MY_ERROR;
    return ret;
}

// Parse the x2APIC_ID_t into SMT, core and package ID.
// http://software.intel.com/en-us/articles/intel-64-architecture-processor-topology-enumeration
void
parse_apic_id(cpuid_info_t info_l0, cpuid_info_t info_l1, APIC_ID_t *my_id){

    // Get the SMT ID
    unsigned int smt_mask_width = info_l0.eax & 0x1f;
    unsigned int smt_mask = ~((-1) << smt_mask_width);
    my_id->smt_id = info_l0.edx & smt_mask;

    // Get the core ID
    unsigned int core_mask_width = info_l1.eax & 0x1f;
    unsigned int core_mask = (~((-1) << core_mask_width ) ) ^ smt_mask;
    my_id->core_id = (info_l1.edx & core_mask) >> smt_mask_width;

    // Get the package ID
    unsigned int pkg_mask = (-1) << core_mask_width;
    my_id->pkg_id = (info_l1.edx & pkg_mask) >> core_mask_width;
}



// For documentation, see:
// http://software.intel.com/en-us/articles/intel-64-architecture-processor-topology-enumeration
int
build_topology() {

    int err;
    unsigned int i,j;
    unsigned int max_pkg = 0;
    unsigned int num_core_threads = 0; // number of physical threads per core
    unsigned int num_pkg_threads = 0;  // number of physical threads per package
    unsigned int num_pkg_cores = 0;  // number of cores per package
    
    os_cpu_count = sysconf(_SC_NPROCESSORS_CONF);

    // Construct an os map: os_map[APIC_ID ... APIC_ID]
    os_map = (APIC_ID_t *) malloc(os_cpu_count * sizeof(APIC_ID_t));

    for(i=0; i < os_cpu_count; i++){

        err = bind_context(i);
        cpuid_info_t info_l0 = get_processor_topology(0);
        cpuid_info_t info_l1 = get_processor_topology(1);

        os_map[i].os_id = i;
        parse_apic_id(info_l0, info_l1, &os_map[i]);

        num_core_threads = info_l0.ebx & 0xffff;
        num_pkg_threads = info_l1.ebx & 0xffff;

        if(os_map[i].pkg_id > max_pkg)
            max_pkg = os_map[i].pkg_id;

        //printf("smt_id: %u core_id: %u pkg_id: %u os_id: %u\n",
        //    os_map[i].smt_id, os_map[i].core_id, os_map[i].pkg_id, os_map[i].os_id);

    }

    num_pkg_cores = num_pkg_threads / num_core_threads;
    num_nodes = max_pkg + 1;

    return err;
}

/*!
 * \brief Intialize the power_gov library for use.
 *
 * This function must be called before calling any other function from the power_gov library.
 * \return 0 on success, -1 otherwise
 */
int
init_rapl()
{
    int          err = 0;
    unsigned int processor_signature;

    processor_signature = get_processor_signature();
    msr_support_table = (unsigned char*) calloc(MSR_SUPPORT_MASK, sizeof(unsigned char));

    switch (processor_signature) {
    case 0x206a7:                /* SandyBridge client*/
        msr_support_table[MSR_RAPL_POWER_UNIT & MSR_SUPPORT_MASK]          = 1;
        msr_support_table[MSR_RAPL_PKG_POWER_LIMIT & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PKG_ENERGY_STATUS & MSR_SUPPORT_MASK]   = 1;
        msr_support_table[MSR_RAPL_PKG_PERF_STATUS & MSR_SUPPORT_MASK]     = 0;
        msr_support_table[MSR_RAPL_PKG_POWER_INFO & MSR_SUPPORT_MASK]      = 1;
        msr_support_table[MSR_RAPL_DRAM_POWER_LIMIT & MSR_SUPPORT_MASK]    = 0;
        msr_support_table[MSR_RAPL_DRAM_ENERGY_STATUS & MSR_SUPPORT_MASK]  = 0;
        msr_support_table[MSR_RAPL_DRAM_PERF_STATUS & MSR_SUPPORT_MASK]    = 0;
        msr_support_table[MSR_RAPL_DRAM_POWER_INFO & MSR_SUPPORT_MASK]     = 0;
        msr_support_table[MSR_RAPL_PP0_POWER_LIMIT & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PP0_ENERGY_STATUS & MSR_SUPPORT_MASK]   = 1;
        msr_support_table[MSR_RAPL_PP0_POLICY & MSR_SUPPORT_MASK]          = 1;
        msr_support_table[MSR_RAPL_PP0_PERF_STATUS & MSR_SUPPORT_MASK]     = 0;
        msr_support_table[MSR_RAPL_PP1_POWER_LIMIT & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PP1_ENERGY_STATUS & MSR_SUPPORT_MASK]   = 1;
        msr_support_table[MSR_RAPL_PP1_POLICY & MSR_SUPPORT_MASK]          = 1;
        break;
    case 0x206d6:                /* SandyBridge server*/
    case 0x206d7:                /* SandyBridge server*/
    case 0x306e4:	         /* IvyBridge server*/
    case 0x306f2:		 /* Haswell server*/
    case 0x406f1:		 /* Broadwell server*/
    case 0x50653:		 /* Skylake server*/
        msr_support_table[MSR_RAPL_POWER_UNIT & MSR_SUPPORT_MASK]          = 1;
        msr_support_table[MSR_RAPL_PKG_POWER_LIMIT & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PKG_ENERGY_STATUS & MSR_SUPPORT_MASK]   = 1;
        msr_support_table[MSR_RAPL_PKG_PERF_STATUS & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PKG_POWER_INFO & MSR_SUPPORT_MASK]      = 1;
        msr_support_table[MSR_RAPL_DRAM_POWER_LIMIT & MSR_SUPPORT_MASK]    = 1;
        msr_support_table[MSR_RAPL_DRAM_ENERGY_STATUS & MSR_SUPPORT_MASK]  = 1;
        msr_support_table[MSR_RAPL_DRAM_PERF_STATUS & MSR_SUPPORT_MASK]    = 1;
        msr_support_table[MSR_RAPL_DRAM_POWER_INFO & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PP0_POWER_LIMIT & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PP0_ENERGY_STATUS & MSR_SUPPORT_MASK]   = 1;
        msr_support_table[MSR_RAPL_PP0_POLICY & MSR_SUPPORT_MASK]          = 0;
        msr_support_table[MSR_RAPL_PP0_PERF_STATUS & MSR_SUPPORT_MASK]     = 1;
        msr_support_table[MSR_RAPL_PP1_POWER_LIMIT & MSR_SUPPORT_MASK]     = 0;
        msr_support_table[MSR_RAPL_PP1_ENERGY_STATUS & MSR_SUPPORT_MASK]   = 0;
        msr_support_table[MSR_RAPL_PP1_POLICY & MSR_SUPPORT_MASK]          = 0;
        break;
    default:
        fprintf(stderr, "RAPL not supported, or machine model (%x) not recognized.\n", processor_signature);
        return MY_ERROR;
    }

    err = read_rapl_units();
    err += build_topology();

    /* 32 is the width of these fields when they are stored */
    MAX_ENERGY_STATUS_JOULES = (double)(RAPL_ENERGY_UNIT * (pow(2, 32) - 1));
    MAX_THROTTLED_TIME_SECONDS = (double)(RAPL_TIME_UNIT * (pow(2, 32) - 1));


    return err;
}

/*!
 * \brief Terminate the power_gov library.
 *
 * Call this function function to cleanup resources and terminate the
 * power_gov library.
 * \return 0 on success
 */
int
terminate_rapl()
{
    unsigned int i;

    if(NULL != os_map)
        free(os_map);

    if(NULL != msr_support_table)
        free(msr_support_table);

    return 0;
}

/*!
 * \brief Check if MSR is supported on this machine.
 * \return 1 if supported, 0 otherwise
 */
unsigned int
is_supported_msr(unsigned int msr)
{
    return (unsigned int)msr_support_table[msr & MSR_SUPPORT_MASK];
}

/*!
 * \brief Check if power domain (PKG, PP0, PP1, DRAM) is supported on this machine.
 *
 * Currently server parts support: PKG, PP0 and DRAM and
 * client parts support PKG, PP0 and PP1.
 *
 * \return 1 if supported, 0 otherwise
 */
unsigned int
is_supported_domain(unsigned int power_domain)
{
    unsigned int supported = 0;

    switch (power_domain) {
    case RAPL_PKG:
        supported = is_supported_msr(MSR_RAPL_PKG_POWER_LIMIT);
        break;
    case RAPL_PP0:
        supported = is_supported_msr(MSR_RAPL_PP0_POWER_LIMIT);
        break;
    case RAPL_PP1:
        supported = is_supported_msr(MSR_RAPL_PP1_POWER_LIMIT);
        break;
    case RAPL_DRAM:
        supported = is_supported_msr(MSR_RAPL_DRAM_POWER_LIMIT);
        break;
    }

    return supported;
}

/*!
 * \brief Get the number of RAPL nodes (package domain) on this machine.
 *
 * Get the number of package power domains, that you can control using RAPL.
 * This is equal to the number of CPU packages in the system.
 *
 * \return number of RAPL nodes.
 */
unsigned int
get_num_rapl_nodes_pkg()
{
    return num_nodes;
}

/*!
 * \brief Get the number of RAPL nodes (pp0 domain) on this machine.
 *
 * Get the number of pp0 (core) power domains, that you can control
 * using RAPL. Currently all the cores on a package belong to the same
 * power domain, so currently this is equal to the number of CPU packages in
 * the system.
 *
 * \return number of RAPL nodes.
 */
unsigned int
get_num_rapl_nodes_pp0()
{
    return num_nodes;
}

/*!
 * \brief Get the number of RAPL nodes (pp1 domain) on this machine.
 *
 * Get the number of pp1(uncore) power domains, that you can control using RAPL.
 * This is equal to the number of CPU packages in the system.
 *
 * \return number of RAPL nodes.
 */
unsigned int
get_num_rapl_nodes_pp1()
{
    return num_nodes;
}

/*!
 * \brief Get the number of RAPL nodes (DRAM domain) on this machine.
 *
 * Get the number of DRAM power domains, that you can control using RAPL.
 * This is equal to the number of CPU packages in the system.
 *
 * \return number of RAPL nodes.
 */
unsigned int
get_num_rapl_nodes_dram()
{
    return num_nodes;
}

unsigned int
pkg_node_to_cpu(unsigned int node)
{
    unsigned int i=0; 
    unsigned int n=0; 
    for(i=0; i < os_cpu_count; i++){
        if(node == os_map[i].pkg_id){
             n = os_map[i].os_id; 
             break; 
        }
    }
    return n; 
}

unsigned int
pp0_node_to_cpu(unsigned int node)
{
    unsigned int i=0; 
    unsigned int n=0; 
    for(i=0; i < os_cpu_count; i++){
        if(node == os_map[i].pkg_id){
             n = os_map[i].os_id; 
             break; 
        }
    }
    return n; 
}

unsigned int
pp1_node_to_cpu(unsigned int node)
{
    unsigned int i=0; 
    unsigned int n=0; 
    for(i=0; i < os_cpu_count; i++){
        if(node == os_map[i].pkg_id){
             n = os_map[i].os_id; 
             break; 
        }
    }
    return n; 
}

unsigned int
dram_node_to_cpu(unsigned int node)
{
    unsigned int i=0; 
    unsigned int n=0; 
    for(i=0; i < os_cpu_count; i++){
        if(node == os_map[i].pkg_id){
             n = os_map[i].os_id; 
             break; 
        }
    }
    return n; 
}

double
convert_to_watts(unsigned int raw)
{
    return RAPL_POWER_UNIT * raw;
}

double
convert_to_joules(unsigned int raw)
{
    return RAPL_ENERGY_UNIT * raw;
}

double
convert_to_seconds(unsigned int raw)
{
    return RAPL_TIME_UNIT * raw;
}

double
convert_from_limit_time_window(unsigned int Y,
                               unsigned int F)
{
    return B2POW(Y) * A_F[F] * RAPL_TIME_UNIT;
}

unsigned int
convert_from_watts(double converted)
{
    return converted / RAPL_POWER_UNIT;
}

unsigned int
compute_Y(unsigned int F,
          double       time)
{
    return (log((double)(time / RAPL_TIME_UNIT)) - A_LNF[F]) / LN2;
}

void
convert_to_limit_time_window(double        time,
                             unsigned int *Y,
                             unsigned int *F)
{
    unsigned int current_Y = 0;
    unsigned int current_F = 0;
    double       current_time = 0.0;
    double       current_delta = 0.0;
    double       delta = 2147483648.0;
    for (current_F = 0; current_F < 4; ++current_F) {
        current_Y = compute_Y(current_F, time);
        current_time = convert_from_limit_time_window(current_Y, current_F);
        current_delta = time - current_time;
        if (current_delta >= 0 && current_delta < delta) {
            delta = current_delta;
            *F = current_F;
            *Y = current_Y;
        }
    }
}

int
get_rapl_unit_multiplier(unsigned int            cpu,
                         rapl_unit_multiplier_t *unit_obj)
{
    int                        err = 0;
    uint64_t                   msr;
    rapl_unit_multiplier_msr_t unit_msr;

    err = !is_supported_msr(MSR_RAPL_POWER_UNIT);
    if (!err) {
        err = read_msr(cpu, MSR_RAPL_POWER_UNIT, &msr);
    }
    if (!err) {
        unit_msr = *(rapl_unit_multiplier_msr_t *)&msr;

        unit_obj->time = 1.0 / (double)(B2POW(unit_msr.time));
        unit_obj->energy = 1.0 / (double)(B2POW(unit_msr.energy));
        unit_obj->power = 1.0 / (double)(B2POW(unit_msr.power));
    }

    return err;
}

/* Common methods (should not be interfaced directly) */

int
get_rapl_power_limit_control(unsigned int                cpu,
                             unsigned int                msr_address,
                             rapl_power_limit_control_t *domain_obj)
{
    int                            err = 0;
    uint64_t                       msr;
    rapl_power_limit_control_msr_t domain_msr;

    err = !is_supported_msr(msr_address);
    if (!err) {
        err = read_msr(cpu, msr_address, &msr);
    }

    if (!err) {
        domain_msr = *(rapl_power_limit_control_msr_t *)&msr;

        domain_obj->power_limit_watts = convert_to_watts(domain_msr.power_limit);
        domain_obj->limit_time_window_seconds = convert_from_limit_time_window(domain_msr.limit_time_window_y,
                                                domain_msr.limit_time_window_f);
        domain_obj->limit_enabled = domain_msr.limit_enabled;
        domain_obj->clamp_enabled = domain_msr.clamp_enabled;
        domain_obj->lock_enabled = domain_msr.lock_enabled;
    }

    return err;
}

int
get_total_energy_consumed(unsigned int  cpu,
                          unsigned int  msr_address,
                          double       *total_energy_consumed_joules)
{
    int                 err = 0;
    uint64_t            msr;
    energy_status_msr_t domain_msr;

    err = !is_supported_msr(msr_address);
    if (!err) {
        err = read_msr(cpu, msr_address, &msr);
    }

    if(!err) {
        domain_msr = *(energy_status_msr_t *)&msr;

        *total_energy_consumed_joules = convert_to_joules(domain_msr.total_energy_consumed);
    }

    return err;
}

int
get_rapl_parameters(unsigned int       cpu,
                    unsigned int       msr_address,
                    rapl_parameters_t *domain_obj)
{
    int                   err = 0;
    uint64_t              msr;
    rapl_parameters_msr_t domain_msr;

    err = !is_supported_msr(msr_address);
    if (!err) {
        err = read_msr(cpu, msr_address, &msr);
    }

    if (!err) {
        domain_msr = *(rapl_parameters_msr_t *)&msr;

        domain_obj->thermal_spec_power_watts = convert_to_watts(domain_msr.thermal_spec_power);
        domain_obj->minimum_power_watts = convert_to_watts(domain_msr.minimum_power);
        domain_obj->maximum_power_watts = convert_to_watts(domain_msr.maximum_power);
        domain_obj->maximum_limit_time_window_seconds = convert_to_seconds(domain_msr.maximum_limit_time_window);
    }

    return err;
}

int
get_accumulated_throttled_time(unsigned int  cpu,
                               unsigned int  msr_address,
                               double       *accumulated_throttled_time_seconds)
{
    int                                 err = 0;
    uint64_t                            msr;
    performance_throttling_status_msr_t domain_msr;

    err = !is_supported_msr(msr_address);
    if (!err) {
        err = read_msr(cpu, msr_address, &msr);
    }

    if (!err) {
        domain_msr = *(performance_throttling_status_msr_t *)&msr;

        *accumulated_throttled_time_seconds = convert_to_seconds(domain_msr.accumulated_throttled_time);
    }

    return err;
}

int
get_balance_policy(unsigned int  cpu,
                   unsigned int  msr_address,
                   unsigned int *priority_level)
{
    int                  err = 0;
    uint64_t             msr;
    balance_policy_msr_t domain_msr;

    err = !is_supported_msr(msr_address);
    if (!err) {
        err = read_msr(cpu, msr_address, &msr);
    }

    if(!err) {
        domain_msr = *(balance_policy_msr_t *)&msr;

        *priority_level = domain_msr.priority_level;
    }

    return err;
}

int
set_rapl_power_limit_control(unsigned int                cpu,
                             unsigned int                msr_address,
                             rapl_power_limit_control_t *domain_obj)
{
    int                            err = 0;
    uint64_t                       msr;
    rapl_power_limit_control_msr_t domain_msr;

    int y;
    int f;

    err = !is_supported_msr(msr_address);
    if (!err) {
        err = read_msr(cpu, msr_address, &msr);
    }

    if (!err) {
        domain_msr = *(rapl_power_limit_control_msr_t *)&msr;

        domain_msr.power_limit = convert_from_watts(domain_obj->power_limit_watts);
        domain_msr.limit_enabled = domain_obj->limit_enabled;
        domain_msr.clamp_enabled = domain_obj->clamp_enabled;
        convert_to_limit_time_window(domain_obj->limit_time_window_seconds, &y, &f);
        domain_msr.limit_time_window_y = y;
        domain_msr.limit_time_window_f = f;
        domain_msr.lock_enabled = domain_obj->lock_enabled;

        msr = *(uint64_t *)&domain_msr;
        err = write_msr(cpu, msr_address, msr);
    }

    return err;
}

int
set_balance_policy(unsigned int cpu,
                   unsigned int msr_address,
                   unsigned int priority_level)
{
    int                  err = 0;
    uint64_t             msr;
    balance_policy_msr_t domain_msr;

    err = !is_supported_msr(msr_address);
    if (!err) {
        err = read_msr(cpu, msr_address, &msr);
    }

    if(!err) {
        domain_msr = *(balance_policy_msr_t *)&msr;

        domain_msr.priority_level = priority_level;

        msr = *(uint64_t *)&domain_msr;
        err = write_msr(cpu, msr_address, msr);
    }

    return err;
}


/* Interface */

/* PKG */

/*!
 * \brief Get a pointer to the RAPL PKG power-limit control register (pkg_rapl_power_limit_control_t).
 *
 * Use the RAPL PKG power-limit control register in order to define power limiting
 * policies on the package power domain.
 * Modify the components of pkg_rapl_power_limit_control_t in order to describe your
 * power limiting policy. Then enforce your new policy using set_pkg_rapl_power_limit_control.
 * At minimum, you should set:
 * - power_limit_watts_1, the power limit to enforce.
 * - limit_enabled_1, enable/disable the power limit.
 * - clamp_enabled_1, when set RAPL is able to overwrite OS requested frequency (full RAPL control).
 *
 * Optionally, you can tune:
 * - limit_time_window_seconds_1, the time slice granularity over which RAPL enforces the power limit
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pkg_rapl_power_limit_control(unsigned int                    node,
                                 pkg_rapl_power_limit_control_t *pkg_obj)
{
    int                                err = 0;
    uint64_t                           msr;
    unsigned int cpu = pkg_node_to_cpu(node);
    pkg_rapl_power_limit_control_msr_t pkg_msr;

    err = !is_supported_msr(MSR_RAPL_PKG_POWER_LIMIT);
    if (!err) {
        err = read_msr(cpu, MSR_RAPL_PKG_POWER_LIMIT, &msr);
    }

    if (!err) {
        pkg_msr = *(pkg_rapl_power_limit_control_msr_t *)&msr;

        pkg_obj->power_limit_watts_1 = convert_to_watts(pkg_msr.power_limit_1);
        pkg_obj->limit_time_window_seconds_1 = convert_from_limit_time_window(pkg_msr.limit_time_window_y_1, pkg_msr.limit_time_window_f_1);
        pkg_obj->limit_enabled_1 = pkg_msr.limit_enabled_1;
        pkg_obj->clamp_enabled_1 = pkg_msr.clamp_enabled_1;
        pkg_obj->power_limit_watts_2 = convert_to_watts(pkg_msr.power_limit_2);
        pkg_obj->limit_time_window_seconds_2 = convert_from_limit_time_window(pkg_msr.limit_time_window_y_2, pkg_msr.limit_time_window_f_2);
        pkg_obj->limit_enabled_2 = pkg_msr.limit_enabled_2;
        pkg_obj->clamp_enabled_2 = pkg_msr.clamp_enabled_2;
        pkg_obj->lock_enabled = pkg_msr.lock_enabled;
    }

    return err;
}

/*!
 * \brief Get a pointer to the RAPL PKG energy consumed register.
 *
 * This read-only register provides energy consumed in joules
 * for the package power domain since the last machine reboot (or energy register wraparound)
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pkg_total_energy_consumed(unsigned int  node,
                              double       *total_energy_consumed_joules)
{
    unsigned int cpu = pkg_node_to_cpu(node);
    return get_total_energy_consumed(cpu, MSR_RAPL_PKG_ENERGY_STATUS, total_energy_consumed_joules);
}

/*!
 * \brief Get a pointer to the RAPL PKG power info register
 *
 * This read-only register provides information about
 * the max/min power limiting settings available on the machine.
 * This register is defined in the pkg_rapl_parameters_t data structure.
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pkg_rapl_parameters(unsigned int           node,
                        pkg_rapl_parameters_t *pkg_obj)
{
    unsigned int cpu = pkg_node_to_cpu(node);
    return get_rapl_parameters(cpu, MSR_RAPL_PKG_POWER_INFO, (rapl_parameters_t*)pkg_obj);
}

/*!
 * \brief Get a pointer to the RAPL PKG throttled time register
 *
 * This read-only register provides information about the amount of time,
 * that a RAPL power limiting policy throttled processor speed in order
 * to prevent a package power limit from being violated.
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pkg_accumulated_throttled_time(unsigned int  node,
                                   double       *accumulated_throttled_time_seconds)
{
    unsigned int cpu = pkg_node_to_cpu(node);
    return get_accumulated_throttled_time(cpu, MSR_RAPL_PKG_PERF_STATUS, accumulated_throttled_time_seconds);
}

/*!
 * \brief Write the RAPL PKG power-limit control register (pkg_rapl_power_limit_control_t).
 *
 * Write the RAPL PKG power-limit control register in order to define power limiting
 * policies on the package power domain.
 *
 * \return 0 on success, -1 otherwise
 */
int
set_pkg_rapl_power_limit_control(unsigned int                    node,
                                 pkg_rapl_power_limit_control_t *pkg_obj)
{
    int                                err = 0;
    uint64_t                           msr;
    unsigned int cpu = pkg_node_to_cpu(node);
    pkg_rapl_power_limit_control_msr_t pkg_msr;

    int y;
    int f;

    err = !is_supported_msr(MSR_RAPL_PKG_POWER_LIMIT);
    if (!err) {
        err = read_msr(cpu, MSR_RAPL_PKG_POWER_LIMIT, &msr);
    }

    if(!err) {
        pkg_msr = *(pkg_rapl_power_limit_control_msr_t *)&msr;

        pkg_msr.power_limit_1 = convert_from_watts(pkg_obj->power_limit_watts_1);
        pkg_msr.limit_enabled_1 = pkg_obj->limit_enabled_1;
        pkg_msr.clamp_enabled_1 = pkg_obj->clamp_enabled_1;
        convert_to_limit_time_window(pkg_obj->limit_time_window_seconds_1, &y, &f);
        pkg_msr.limit_time_window_y_1 = y;
        pkg_msr.limit_time_window_f_1 = f;
        pkg_msr.power_limit_2 = convert_from_watts(pkg_obj->power_limit_watts_2);
        pkg_msr.limit_enabled_2 = pkg_obj->limit_enabled_2;
        pkg_msr.clamp_enabled_2 = pkg_obj->clamp_enabled_2;
        convert_to_limit_time_window(pkg_obj->limit_time_window_seconds_2, &y, &f);
        pkg_msr.limit_time_window_y_2 = y;
        pkg_msr.limit_time_window_f_2 = f;
        pkg_msr.lock_enabled = pkg_obj->lock_enabled;

        msr = *(uint64_t *)&pkg_msr;
        err = write_msr(cpu, MSR_RAPL_PKG_POWER_LIMIT, msr);
    }

    return err;
}


/* DRAM */

/*!
 * \brief Get a pointer to the RAPL DRAM power-limit control register (dram_rapl_power_limit_control_t).
 *
 * (Server parts only)
 *
 * Use the RAPL DRAM power-limit control register in order to define power limiting
 * policies on the DRAM power domain.
 * Modify the components of dram_rapl_power_limit_control_t in order to describe your
 * power limiting policy. Then enforce your new policy using set_dram_rapl_power_limit_control
 *  At minimum, you should set:
 * - power_limit_watts, the power limit to enforce.
 * - limit_enabled, enable/disable the power limit.
 *
 * Optionally, you can tune:
 * - limit_time_window_seconds, the time slice granularity over which RAPL enforces the power limit
 *
 * \return 0 on success, -1 otherwise
 */
int
get_dram_rapl_power_limit_control(unsigned int                     node,
                                  dram_rapl_power_limit_control_t *dram_obj)
{
    unsigned int cpu = dram_node_to_cpu(node);
    return get_rapl_power_limit_control(cpu, MSR_RAPL_DRAM_POWER_LIMIT, (rapl_power_limit_control_t*)dram_obj);
}

/*!
 * \brief Get a pointer to the RAPL DRAM energy consumed register.
 *
 * (Server parts only)
 *
 * This read-only register provides energy consumed in joules
 * for the DRAM power domain since the last machine reboot (or energy register wraparound)
 *
 * \return 0 on success, -1 otherwise
 */
int
get_dram_total_energy_consumed(unsigned int  node,
                               double       *total_energy_consumed_joules)
{
    unsigned int cpu = dram_node_to_cpu(node);
    return get_total_energy_consumed(cpu, MSR_RAPL_DRAM_ENERGY_STATUS, total_energy_consumed_joules);
}

/*!
 * \brief Get a pointer to the RAPL DRAM power info register
 *
 * (Server parts only)
 *
 * This read-only register provides information about
 * the max/min power limiting settings available on the machine.
 * This register is defined in the dram_rapl_parameters_t data structure.
 *
 * \return 0 on success, -1 otherwise
 */
int
get_dram_rapl_parameters(unsigned int            node,
                         dram_rapl_parameters_t *dram_obj)
{
    unsigned int cpu = dram_node_to_cpu(node);
    return get_rapl_parameters(cpu, MSR_RAPL_DRAM_POWER_INFO, (rapl_parameters_t*)dram_obj);
}

/*!
 * \brief Get a pointer to the RAPL DRAM throttled time register
 *
 * (Server parts only)
 *
 * This read-only register provides information about the amount of time,
 * that a RAPL power limiting policy throttled DRAM bandwidth in order
 * to prevent a DRAM power limit from being violated.
 *
 * \return 0 on success, -1 otherwise
 */
int
get_dram_accumulated_throttled_time(unsigned int  node,
                                    double       *accumulated_throttled_time_seconds)
{
    unsigned int cpu = dram_node_to_cpu(node);
    return get_accumulated_throttled_time(cpu, MSR_RAPL_DRAM_PERF_STATUS, accumulated_throttled_time_seconds);
}

/*!
 * \brief Write the RAPL DRAM power-limit control register (dram_rapl_power_limit_control_t).
 *
 * (Server parts only)
 *
 * Write the RAPL DRAM power-limit control register in order to define power limiting
 * policies on the DRAM power domain.
 *
 * \return 0 on success, -1 otherwise
 */
int
set_dram_rapl_power_limit_control(unsigned int                     node,
                                  dram_rapl_power_limit_control_t *dram_obj)
{
    unsigned int cpu = dram_node_to_cpu(node);
    return set_rapl_power_limit_control(cpu, MSR_RAPL_DRAM_POWER_LIMIT, (rapl_power_limit_control_t*)dram_obj);
}


/* PP0 */

/*!
 * \brief Get a pointer to the RAPL PP0 power-limit control register (pp0_rapl_power_limit_control_t).
 *
 * Use the RAPL PP0 power-limit control register in order to define power limiting
 * policies on the PP0 (core) power domain.
 * Modify the components of pp0_rapl_power_limit_control_t in order to describe your
 * power limiting policy. Then enforce your new policy using set_pp0_rapl_power_limit_control
 *  At minimum, you should set:
 * - power_limit_watts, the power limit to enforce.
 * - limit_enabled, enable/disable the power limit.
 *
 * Optionally, you can tune:
 * - limit_time_window_seconds, the time slice granularity over which RAPL enforces the power limit
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pp0_rapl_power_limit_control(unsigned int                    node,
                                 pp0_rapl_power_limit_control_t *pp0_obj)
{
    unsigned int cpu = pp0_node_to_cpu(node);
    return get_rapl_power_limit_control(cpu, MSR_RAPL_PP0_POWER_LIMIT, (rapl_power_limit_control_t*)pp0_obj);
}

/*!
 * \brief Get a pointer to the RAPL PP0 energy consumed register.
 *
 * This read-only register provides energy consumed in joules
 * for the PP0 (core) power domain since the last machine reboot (or energy register wraparound)
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pp0_total_energy_consumed(unsigned int  node,
                              double       *total_energy_consumed_joules)
{
    unsigned int cpu = pp0_node_to_cpu(node);
    return get_total_energy_consumed(cpu, MSR_RAPL_PP0_ENERGY_STATUS, total_energy_consumed_joules);
}

/*!
 * \brief Get a pointer to the RAPL PP0 priority level register
 *
 * (Client parts only)
 *
 * Use the RAPL PP0 priority level register in order to provide an input
 * to the power budgeting algorithm on how to distribute power between the
 * PP0 (core) place and PP1 (uncore - graphics). The default value gives
 * priority to the uncore power plane. After modifying the register
 * enforce your setting using set_pp0_balance_policy.
 * The value 31 is considered highest priority and 0 lowest.
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pp0_balance_policy(unsigned int  node,
                       unsigned int *priority_level)
{
    unsigned int cpu = pp0_node_to_cpu(node);
    return get_balance_policy(cpu, MSR_RAPL_PP0_POLICY, priority_level);
}

/*!
 * \brief Get a pointer to the RAPL PP0 throttled time register
 *
 * This read-only register provides information about the amount of time,
 * that a RAPL power limiting policy throttled processor speed in order
 * to prevent a PP0 (core) power limit from being violated.
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pp0_accumulated_throttled_time(unsigned int  node,
                                   double       *accumulated_throttled_time_seconds)
{
    unsigned int cpu = pp0_node_to_cpu(node);
    return get_accumulated_throttled_time(cpu, MSR_RAPL_PP0_PERF_STATUS, accumulated_throttled_time_seconds);
}

/*!
 * \brief Write the RAPL PP0 power-limit control register (pp0_rapl_power_limit_control_t).
 *
 * Write the RAPL PP0 power-limit control register in order to define power limiting
 * policies on the PP0 power domain.
 *
 * \return 0 on success, -1 otherwise
 */
int
set_pp0_rapl_power_limit_control(unsigned int                    node,
                                 pp0_rapl_power_limit_control_t *pp0_obj)
{
    unsigned int cpu = pp0_node_to_cpu(node);
    return set_rapl_power_limit_control(cpu, MSR_RAPL_PP0_POWER_LIMIT, (rapl_power_limit_control_t*)pp0_obj);
}

/*!
 * \brief Write to the RAPL PP0 priority level register
 *
 * (Client parts only)
 *
 * Use the RAPL PP0 priority level register in order to provide an input
 * to the power budgeting algorithm on how to distribute power between the
 * PP0 (core) place and PP1 (uncore - graphics). The default value gives
 * priority to the uncore power plane.
 *
 * \return 0 on success, -1 otherwise
 */
int
set_pp0_balance_policy(unsigned int node,
                       unsigned int priority_level)
{
    unsigned int cpu = pp0_node_to_cpu(node);
    return set_balance_policy(cpu, MSR_RAPL_PP0_POLICY, priority_level);
}


/* PP1 */

/*!
 * \brief Get a pointer to the RAPL PP1 power-limit control register (pp1_rapl_power_limit_control_t).
 *
 * (Client parts only)
 *
 * Use the RAPL PP1 power-limit control register in order to define power limiting
 * policies on the PP1 (uncore) power domain.
 * Modify the components of pp1_rapl_power_limit_control_t in order to describe your
 * power limiting policy. Then enforce your new policy using set_pp1_rapl_power_limit_control
 *  At minimum, you should set:
 * - power_limit_watts, the power limit to enforce.
 * - limit_enabled, enable/disable the power limit.
 *
 * Optionally, you can tune:
 * - limit_time_window_seconds, the time slice granularity over which RAPL enforces the power limit
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pp1_rapl_power_limit_control(unsigned int                    node,
                                 pp1_rapl_power_limit_control_t *pp1_obj)
{
    unsigned int cpu = pp1_node_to_cpu(node);
    return get_rapl_power_limit_control(cpu, MSR_RAPL_PP1_POWER_LIMIT, (rapl_power_limit_control_t*)pp1_obj);
}

/*!
 * \brief Get a pointer to the RAPL PP1 energy consumed register.
 *
 * (Client parts only)
 *
 * This read-only register provides energy consumed in joules
 * for the PP1 (uncore) power domain since the last machine reboot (or energy register wraparound)
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pp1_total_energy_consumed(unsigned int  node,
                              double       *total_energy_consumed_joules)
{
    unsigned int cpu = pp1_node_to_cpu(node);
    return get_total_energy_consumed(cpu, MSR_RAPL_PP1_ENERGY_STATUS, total_energy_consumed_joules);
}

/*!
 * \brief Get a pointer to the RAPL PP1 priority level register
 *
 * (Client parts only)
 *
 * Use the RAPL PP1 priority level register in order to provide an input
 * to the power budgeting algorithm on how to distribute power between the
 * PP0 (core) place and PP1 (uncore - graphics). The default value gives
 * priority to the uncore power plane. After modifying the register
 * enforce your setting using set_pp1_balance_policy.
 * The value 31 is considered highest priority and 0 lowest.
 *
 * \return 0 on success, -1 otherwise
 */
int
get_pp1_balance_policy(unsigned int  node,
                       unsigned int *priority_level)
{
    unsigned int cpu = pp1_node_to_cpu(node);
    return get_balance_policy(cpu, MSR_RAPL_PP1_POLICY, priority_level);
}

/*!
 * \brief Write the RAPL PP1 power-limit control register (pp1_rapl_power_limit_control_t).
 *
 * (Client parts only)
 *
 * Write the RAPL PP1 power-limit control register in order to define power limiting
 * policies on the PP1 (uncore) power domain.
 *
 * \return 0 on success, -1 otherwise
 */
int
set_pp1_rapl_power_limit_control(unsigned int                    node,
                                 pp1_rapl_power_limit_control_t *pp1_obj)
{
    unsigned int cpu = pp1_node_to_cpu(node);
    return set_rapl_power_limit_control(cpu, MSR_RAPL_PP1_POWER_LIMIT, (rapl_power_limit_control_t*)pp1_obj);
}

/*!
 * \brief Write to the RAPL PP1 priority level register
 *
 * (Client parts only)
 *
 * Use the RAPL PP1 priority level register in order to provide an input
 * to the power budgeting algorithm on how to distribute power between the
 * PP0 (core) place and PP1 (uncore - graphics). The default value gives
 * priority to the uncore power plane.
 *
 * \return 0 on success, -1 otherwise
 */
int
set_pp1_balance_policy(unsigned int node,
                       unsigned int priority_level)
{
    unsigned int cpu = pp1_node_to_cpu(node);
    return set_balance_policy(cpu, MSR_RAPL_PP1_POLICY, priority_level);
}

/* Utilities */

int
read_rapl_units()
{
    int                    err = 0;
    rapl_unit_multiplier_t unit_multiplier;

    err = get_rapl_unit_multiplier(0, &unit_multiplier);
    if (!err) {
        RAPL_TIME_UNIT = unit_multiplier.time;
        RAPL_ENERGY_UNIT = unit_multiplier.energy;
        RAPL_POWER_UNIT = unit_multiplier.power;
    }

    return err;
}

