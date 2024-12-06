/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_BN_FIN_HPP
#define GUARD_BN_FIN_HPP

#include "error.hpp"
#include "fin.hpp"
#include "random.hpp"
#include "rocrand_wrapper.hpp"
#include "gpuMemTensor.hpp"
#include "random_test.hpp"
#include "tensor_holder.hpp"

#include <miopen/execution_context.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/miopen.h>
#include <miopen/batchnorm/problem_description.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/batchnorm/invoke_params.hpp>
#include <miopen/batchnorm/solvers.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/solver.hpp>
#include <miopen/solver_id.hpp>
#include <miopen/driver_arguments.hpp>
#include <miopen/tensor.hpp>
#include <miopen/fin/fin_interface.hpp>

#include <nlohmann/json.hpp>

#define EPSILON 1e-3

namespace fs = miopen::fs;
namespace fin {

template <typename TInput,
          typename Tref,
          typename TAcc       = TInput,
          typename TScaleBias = TInput,
          typename TOut       = TInput>
class BNFin : public BaseFin
{
    public:
    BNFin() : BaseFin() {}
    BNFin(json _job) : BaseFin(), job(_job)
    {
        if(job.contains("config"))
            PrepBatchNorm();
    }

    void PrepBatchNorm()
    {
        BaseFin::VerifyDevProps(job["arch"], job["num_cu"]);
        command         = job["config"];
        command["bias"] = 0;
        SetBNDescriptor();
        isFwdTrain = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 1);
        isFwdInfer = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 2);
        isBwd      = (job["direction"].get<int>() == 0 || job["direction"].get<int>() & 4);
    }

    // Getters and setters
    std::vector<int> GetInputTensorLengths(); // checked
    std::vector<int> GetBiasTensorLengths();
    int SetBNDescriptor();
    miopen::debug::BatchNormDirection_t GetDirection() const;

    int ProcessStep(const std::string& step_name) override;

    // Steps
    int TestApplicability();
    int GetandSetData();
    std::vector<miopen::solver::ConvSolution> GetBNSolutions(miopen::ExecutionContext& ctx);
    miopen::batchnorm::ProblemDescription GetProblemDescription();
    auto GetAlgorithm();

    int MIOpenCompile(TuningOp tuning_op);

    float PerfTune(const miopen::Handle& h,
                   const miopen::solver::ConvSolution& solution,
                   miopen::PerformanceDb& db,
                   miopen::ExecutionContext& perf_ctx);
    int MIOpenEval(TuningOp tuning_op);

    float FindTune(const miopen::Handle& h, const miopen::solver::ConvSolution& solution);
    const miopen::AnyInvokeParams GetInvokeCtx();

    // Utility functions
    auto GetFwdTrainSolvers();
    auto GetFwdInferSolvers();
    auto GetBwdSolvers();

    json command;
    json job;

    miopenBatchNormMode_t bn_mode;
    miopenActivationMode_t activ_mode = miopenActivationRELU;
    std::vector<std::string> steps_processed;
    bool saveMeanVar        = false;                      // checked
    bool keepRunningMeanVar = false;                      // checked
    Tref epsilon            = static_cast<Tref>(EPSILON); // checked
    double expAvgFactor     = 1.0;
    bool isDepthSpecified   = false;
    bool isFwdTrain         = true;
    bool isFwdInfer         = false;
    bool isBwd              = false;

    tensor<TInput> workspace;
    GpumemTensor<TInput> in;
    GpumemTensor<TInput> out;
    GpumemTensor<Tref> out_ref;

    // forward
    GpumemTensor<TScaleBias> scale;
    GpumemTensor<TScaleBias> bias;

    // forward inference
    GpumemTensor<TAcc> estMean;
    GpumemTensor<TAcc> estVariance;

    GpumemTensor<TAcc> savedMean;
    Tensor<Tref> savedMean_ref;

    // forward training
    GpumemTensor<TAcc> savedVariance;
    GpumemTensor<TAcc> runMean;
    GpumemTensor<TAcc> runVariance;
    // ref
    Tensor<Tref> savedVariance_ref;
    Tensor<Tref> runMean_ref;
    Tensor<Tref> runVariance_ref;

    // backward needed different type for bwd.
    GpumemTensor<TOut> out_bwd;

    GpumemTensor<TScaleBias> bnScale;
    GpumemTensor<TAcc> dScale;
    GpumemTensor<TAcc> dBias;

    GpumemTensor<TAcc> savedInvVar;
    GpumemTensor<TOut> dy;

    Tensor<Tref> dBias_ref;
    Tensor<Tref> dScale_ref;

    // for backward
    // Tensor<TInput, Tcpu> dyInputTensor;
    // Tensor<TInput, Tcpu> dxOutputTensor;

    // Tref maxval;
    miopenTensorLayout_t bn_layout;
};

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
miopen::debug::BatchNormDirection_t
BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetDirection() const
{
    return isFwdTrain ? miopen::debug::BatchNormDirection_t::ForwardTraining
                      : (isFwdInfer ? miopen::debug::BatchNormDirection_t::ForwardInference
                                    : miopen::debug::BatchNormDirection_t::Backward);
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::TestApplicability()
{
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif

    auto& handle = GetHandle();
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ExecutionContext(&handle);
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "to test applicability");
#endif
    ctx.SetStream(&handle);

    std::vector<std::string> app_solvers;

    for(const auto& sln : GetBNSolutions(ctx))

    {
        std::cerr << sln.solver_id << std::endl;
        if(!sln.invoker_factory)
        {
            MIOPEN_THROW(miopenStatusInternalError, "Invoker missing in solver " + sln.solver_id);
        }
        app_solvers.push_back(sln.solver_id);
    }
    for(const auto& elem : app_solvers)
    {
        std::cerr << elem << std::endl;
    }

    output["applicable_solvers"] = app_solvers;
    return 0;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetandSetData()
{

    SetBNDescriptor();
    auto in_len = GetInputTensorLengths();

    in.AllocOnHost(Tensor<TInput>{bn_layout, in_len});

    auto derivedBnDesc = miopen::TensorDescriptor{};
    miopen::DeriveBNTensorDescriptor(derivedBnDesc, in.GetTensor().desc, bn_mode);

    if(isFwdInfer || isFwdTrain)
    {
        out.AllocOnHost(Tensor<TInput>{bn_layout, in_len});
        scale.AllocOnHost(Tensor<TInput>{bn_layout, derivedBnDesc.GetLengths()});
        bias.AllocOnHost(Tensor<TInput>{bn_layout, derivedBnDesc.GetLengths()});

        for(int i = 0; i < scale.GetVector().size(); i++)
        {
            scale.GetVector()[i] = prng::gen_canonical<TInput>();
            bias.GetVector()[i]  = prng::gen_canonical<TInput>();
        }
    }
    if(isFwdInfer)
    {
        estMean.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        estVariance.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});

        auto gen_value_emean = [](auto...) {
            return prng::gen_descreet_uniform_sign<TAcc>(1e-2, 100);
        };
        estMean.InitHostData(estMean.GetTensor().desc.GetElementSize(), true, gen_value_emean);
    }
    else if(isFwdTrain)
    {
        savedMean.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        savedVariance.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        runMean.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        runVariance.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});

        for(int i = 0; i < runVariance.GetVector().size(); i++)
        {
            runMean.GetVector()[i]     = prng::gen_canonical<TAcc>();
            runVariance.GetVector()[i] = prng::gen_canonical<TAcc>();
        }
    }
    else if(isBwd)
    {

        out_bwd.AllocOnHost(Tensor<TOut>{bn_layout, in_len});

        bnScale.AllocOnHost(Tensor<TScaleBias>{bn_layout, derivedBnDesc.GetLengths()});
        dy.AllocOnHost(Tensor<TOut>{bn_layout, in_len});

        auto gen_var_bwd = [](auto...) {
            return static_cast<TOut>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
        dy.InitHostData(dy.GetTensor().desc.GetElementSize(), true, gen_var_bwd);

        dScale.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        dBias.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        savedMean.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});
        savedInvVar.AllocOnHost(Tensor<TAcc>{bn_layout, derivedBnDesc.GetLengths()});

        auto gen_value = [](auto...) { return prng::gen_descreet_unsigned<TInput>(1e-2, 100); };
        bnScale.InitHostData(bnScale.GetTensor().desc.GetElementSize(), true, gen_value);

        auto gen_in_var = [](auto...) {
            return static_cast<TAcc>(1e-2 * (prng::gen_0_to_B(100) + 1));
        };
        savedMean.InitHostData(savedMean.GetTensor().desc.GetElementSize(), true, gen_in_var);
        savedInvVar.InitHostData(savedInvVar.GetTensor().desc.GetElementSize(), true, gen_in_var);
    }
    else
    {
        std::cout << "\nUnknown batch norm state!\n";
        exit(EXIT_FAILURE);
    }

    return (0);
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
std::vector<int> BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetInputTensorLengths()
{
    int in_n = command["batchsize"];
    int in_c = command["in_channels"];
    int in_h = command["in_h"];
    int in_w = command["in_w"];
    int in_d = command["in_d"];

    if(command["in_d"] > 1)
    {
        isDepthSpecified = true;
        // NxCxDxHxW -> NxCx(D*H)xW
        return std::vector<int>({in_n, in_c, in_d, in_h, in_w});
    }
    else
    {
        isDepthSpecified = false;
        return std::vector<int>({in_n, in_c, in_h, in_w});
    }
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
std::vector<int> BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetBiasTensorLengths()
{
    int spatial_dim = 2;
    if(command["in_d"] > 1)
    {
        spatial_dim = 3;
    }

    std::vector<int> bias_lens(2 + spatial_dim, 1);

    bias_lens[1] = command["out_channels"];

    return bias_lens;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::ProcessStep(const std::string& step_name)
{
    steps_processed.push_back(step_name);
    if(step_name == "applicability")
        return TestApplicability();
    if(step_name == "miopen_perf_compile")
        return MIOpenCompile(TuningOp::Perf);
    if(step_name == "miopen_perf_eval")
        return MIOpenEval(TuningOp::Perf);
    if(step_name == "miopen_find_compile")
        return MIOpenCompile(TuningOp::Find);
    if(step_name == "miopen_find_eval")
        return MIOpenEval(TuningOp::Find);
    return 0;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::SetBNDescriptor()
{
    // batch norm mode type
    bn_mode = command["mode"] == 0 ? miopenBNPerActivation : miopenBNSpatial;

    // save off mean and variance?
    saveMeanVar = command["save"] == 0 ? false : true;

    // keep running mean and variance
    keepRunningMeanVar = command["run"] == 0 ? false : true;

    // set & sanity check for memory layout
    if(command["layout"] == "NCHW")
    {
        bn_layout = miopenTensorLayout_t::miopenTensorNCHW;
    }
    else if(command["layout"] == "NHWC")
    {
        bn_layout = miopenTensorLayout_t::miopenTensorNHWC;
    }
    else if(command["layout"] == "NCDHW")
    {
        bn_layout = miopenTensorLayout_t::miopenTensorNCDHW;
    }
    else if(command["layout"] == "NDHWC")
    {
        bn_layout = miopenTensorLayout_t::miopenTensorNDHWC;
    }
    else
    {
        throw std::runtime_error("Provided memory layout is : " + std::string(command["layout"]) +
                                 ". Batch norm only support default NCHW, NHWC, NCDHW, NDHWC");
    }

    return miopenStatusSuccess;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
auto BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetFwdTrainSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnFwdTrainingSpatialSingle,
                                           //  solver::batchnorm::BnCKFwdTraining,
                                           miopen::solver::batchnorm::BnFwdTrainingSpatialMultiple,
                                           miopen::solver::batchnorm::BnFwdTrainingPerActivation>{};
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
auto BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetFwdInferSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnFwdInference>{};
    //  miopen::solver::batchnorm::BnCKFwdInference
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
auto BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetBwdSolvers()
{
    return miopen::solver::SolverContainer<miopen::solver::batchnorm::BnBwdTrainingSpatialSingle,
                                           //  miopen::solver::batchnorm::BnCKBwdBackward,
                                           miopen::solver::batchnorm::BnBwdTrainingSpatialMultiple,
                                           miopen::solver::batchnorm::BnBwdTrainingPerActivation>{};
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
miopen::batchnorm::ProblemDescription
BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetProblemDescription()
{
    if(isFwdTrain)
    {
        return miopen::batchnorm::ProblemDescription{bn_mode,
                                                     in.GetTensor().desc,
                                                     out.GetTensor().desc,
                                                     scale.GetTensor().desc,
                                                     bias.GetTensor().desc,
                                                     savedMean.GetTensor().desc,
                                                     savedVariance.GetTensor().desc,
                                                     expAvgFactor,
                                                     epsilon,
                                                     saveMeanVar,         //?
                                                     keepRunningMeanVar}; //?
    }
    else if(isFwdInfer)
    {
        return miopen::batchnorm::ProblemDescription(bn_mode,
                                                     in.GetTensor().desc,
                                                     out.GetTensor().desc,
                                                     scale.GetTensor().desc,
                                                     bias.GetTensor().desc,
                                                     estMean.GetTensor().desc,
                                                     estVariance.GetTensor().desc,
                                                     epsilon);
    }
    else if(isBwd)
    {
        // const auto useSaved = savedMean.GetTensor() != nullptr && savedVariance.GetTensor() !=
        // nullptr; ??
        bool useSaved = 0;
        return miopen::batchnorm::ProblemDescription(bn_mode,
                                                     in.GetTensor().desc,
                                                     dy.GetTensor().desc,
                                                     out_ref.GetTensor().desc,
                                                     scale.GetTensor().desc,
                                                     bias.GetTensor().desc,
                                                     savedMean.GetTensor().desc,
                                                     savedVariance.GetTensor().desc,
                                                     epsilon,
                                                     useSaved);
    }
    else
    {
        throw std::runtime_error("Unable to get solvers for batch norm");
    }
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
std::vector<miopen::solver::ConvSolution>
BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetBNSolutions(miopen::ExecutionContext& ctx)
{
    const auto problem = GetProblemDescription();
    if(isFwdTrain)
    {
        return GetFwdTrainSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else if(isFwdInfer)
    {
        return GetFwdInferSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else if(isBwd)
    {
        return GetBwdSolvers().SearchForSolutions(ctx, problem, 1);
    }
    else
    {
        throw std::runtime_error("Unable to to get solutions for batch norm");
    }
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
auto BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetAlgorithm()
{
    if(isFwdTrain)
    {
        return bn_mode == miopenBNSpatial
                   ? miopen::AlgorithmName{"miopenBatchNormForwardTrainingSpatial"}
                   : miopen::AlgorithmName{"miopenBatchNormForwardTrainingPerActivation"};
    }
    else if(isFwdInfer)
    {
        return miopen::AlgorithmName{"miopenBatchNormalizationForwardInference"};
    }
    else if(isBwd)
    {
        return bn_mode == miopenBNSpatial
                   ? miopen::AlgorithmName{"miopenBatchNormBackwardPropSpatial"}
                   : miopen::AlgorithmName{"miopenBatchNormBackwardPropPerActivation"};
    }
    else
    {
        throw std::runtime_error("Unable to get solvers for batch norm");
    }
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::MIOpenCompile(TuningOp tuning_op)
{
    std::cerr << "MIOpenFinCompile" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
#if MIOPEN_MODE_NOGPU
    GetandSetData();
#else
    throw std::runtime_error(
        "Unable to perform MIOpenCompile MIOpen was not compiled using HIPNOGPU backend");
#endif
    auto& handle = GetHandle();
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ExecutionContext(&handle);
    GetHandle().EnableProfiling(true);
#if MIOPEN_MODE_NOGPU
    BaseFin::InitNoGpuHandle(handle, job["arch"], job["num_cu"]);
#else
    throw std::runtime_error("MIOpen needs to be compiled with the NOGPU backend "
                             "for Batch Norm find_compile");
#endif
    ctx.SetStream(&handle);

    const auto problem        = GetProblemDescription();
    const auto network_config = problem.MakeNetworkConfig();
    output["network_config"]  = network_config;
    std::ostringstream ss;
    problem.Serialize(ss);
    // output["db_key"] = ss.str();
    output["is_winograd_only"] = false;

    json find_result;
    std::cerr << "Job Arch: " << job["arch"]
              << ": Handle Arch: " << handle.GetTargetProperties().Name() << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"]
              << ": Handle Num Cu: " << handle.GetMaxComputeUnits() << std::endl;

    if(job.contains("dynamic_only"))
        ctx.use_dynamic_solutions_only = true;

    auto db = GetDb(ctx);
    json comp_res;

    for(const auto& sln : GetBNSolutions(ctx))
        std::cout << "SLN: " << sln.solver_id << std::endl;

    for(const auto& sln : GetBNSolutions(ctx))
    {
        json res_item;
        res_item["reason"]    = std::string("No solutions: ");
        auto process_solution = [&]() -> bool {
            // remove the user db files
            fs::remove_all(miopen::GetCachePath(false));
            std::cerr << "Processing Solver: " << sln.solver_id << std::endl;
            if((job.contains("solvers") &&
                (std::find(std::begin(job["solvers"]), std::end(job["solvers"]), sln.solver_id) !=
                 std::end(job["solvers"]))) ||
               (!job.contains("solvers")))
            {
                res_item["solver_name"] = sln.solver_id;
                std::cout << sln.solver_id << std::endl;
                std::cout << res_item["solver_name"] << std::endl;
                const auto solver = miopen::fin_interface::GetBatchNormSolver(sln.solver_id);

                if(!solver.IsValid())
                {
                    res_item["reason"] = "Solver not valid";
                    std::cerr << "Skipping invalid solver: " << sln.solver_id << std::endl;
                    return false;
                }

                res_item["algorithm"] = GetAlgorithm();

                if(tuning_op == TuningOp::Perf)
                {
                    std::vector<miopen::solver::KernelInfo> kernels;
                    for(auto&& kernel :
                        sln.construction_params) // cppcheck-suppress useStlAlgorithm
                        kernels.push_back(kernel);
                    std::ignore = miopen::solver::PrecompileKernels(handle, kernels);

                    res_item["kernel_objects"] = BuildJsonKernelList(handle, kernels);
                }
                else if(tuning_op == TuningOp::Find)
                {
                    //  NOTE: how to get params from solution?
                    res_item["params"]    = solver.GetPerfCfgParams(ctx, problem, db);
                    res_item["workspace"] = sln.workspace_sz;
                    res_item["kernel_objects"] =
                        BuildJsonKernelList(handle, sln.construction_params);
                }
                res_item["tunable"] = solver.IsTunable();
                res_item["reason"]  = "Success";
                return true;
            }
            return false;
        };

        auto res = process_solution();

        if(tuning_op == TuningOp::Perf)
            res_item["perf_compiled"] = res;
        if(tuning_op == TuningOp::Find)
            res_item["find_compiled"] = res;
        comp_res.push_back(res_item);
    }

    if(tuning_op == TuningOp::Perf)
        output["miopen_perf_compile_result"] = comp_res;
    if(tuning_op == TuningOp::Find)
        output["miopen_find_compile_result"] = comp_res;
    return 1;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
int BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::MIOpenEval(TuningOp tuning_op)
{
    std::cerr << "MIOpenEval" << std::endl;
    std::cerr << "Processing command: " << command << std::endl;
// Before this step is executed, the following steps should have been evaluated
// alloc_buf only if only timing is required
// alloc_buf, fill_buf and copy_buf_to_device if numerical accuracy would be
// checked ??
#if MIOPEN_MODE_NOGPU
    throw std::runtime_error("Unable to run MIOpenEval, Invalid MIOpen backend: HIPNOGPU");
#endif
    const auto bn_dir = GetDirection();
    // The first arg to the DataInvokeParams changes based on direction
    const auto problem = GetProblemDescription();

    GetHandle().EnableProfiling(true);
    auto& h = GetHandle();
    // cppcheck-suppress unreadVariable
    auto ctx = miopen::ExecutionContext(&h);
    ctx.SetStream(&(h));
    // problem.SetupFloats(ctx); ?? not available in batchnorm::PD

    output["is_winograd_only"] = false;
    const auto network_config  = problem.MakeNetworkConfig();
    output["network_config"]   = network_config;
    std::ostringstream ss;
    problem.Serialize(ss);
    // output["db_key"] = ss.str();

    auto db = GetDb(ctx, problem);
    json eval_result;
    const auto& tgt_props  = h.GetTargetProperties();
    const std::string arch = tgt_props.Name();
    const size_t num_cu    = h.GetMaxComputeUnits();
    std::cerr << "Job Arch: " << job["arch"] << ": Handle Arch: " << arch << std::endl;
    std::cerr << "Job Num CU: " << job["num_cu"] << ": Handle Num Cu: " << num_cu << std::endl;
    bool dynamic_only = false;
    if(job.contains("dynamic_only"))
        ctx.use_dynamic_solutions_only = true;

    std::string comp_res_str;
    if(tuning_op == TuningOp::Perf)
        comp_res_str = "miopen_perf_compile_result";
    else if(tuning_op == TuningOp::Find)
        comp_res_str = "miopen_find_compile_result";

    for(const auto& solution : GetBNSolutions(ctx))
    {
        for(const auto& eval_slv : job[comp_res_str])
        {
            json res_item;
            if(solution.solver_id == eval_slv)
            {
                auto process_solver = [&]() -> bool {
                    // remove the user db files
                    fs::remove_all(miopen::GetCachePath(false));
                    std::cerr << "Processing Solver: " << solution.solver_id << std::endl;
                    res_item["solver_name"] = solution.solver_id;
                    const auto solver =
                        miopen::fin_interface::GetBatchNormSolver(solution.solver_id);
                    res_item["algorithm"] = GetAlgorithm();

                    if(dynamic_only && !solver.IsDynamic())
                    {
                        res_item["reason"] = "Not Dynamic";
                        std::cerr << "Skipping static solver: " << solution.solver_id << std::endl;
                        return false;
                    }

                    // Get the binary
                    std::cerr << "Applicable solver: " << solution.solver_id
                              << ", loading binaries from fin input" << std::endl;
                    if(!LoadJsonKernelList(h, eval_slv["kernel_objects"], res_item))
                        return false;

                    SolutionHasProgram(h, solution);

                    std::cerr << "Checking workspace size" << std::endl;
                    if(solution.workspace_sz > workspace.desc.GetNumBytes())
                    {
                        std::cerr << "Allocating " << solution.workspace_sz
                                  << " bytes for workspace" << std::endl;
                        workspace = tensor<TInput>{q,
                                                   std::vector<size_t>{static_cast<size_t>(
                                                       solution.workspace_sz / sizeof(TInput))},
                                                   false,
                                                   false};
                        workspace.AllocateBuffers();
                    }
                    if(!solution.invoker_factory)
                    {
                        std::cerr << "Invoker not implemeted" << std::endl;
                        res_item["reason"] = "Invoker not implemented";
                        return false;
                    }
                    try
                    {
                        float kernel_time = -1;
                        if(tuning_op == TuningOp::Perf)
                            kernel_time = PerfTune(h, solution, db, ctx);
                        else if(tuning_op == TuningOp::Find)
                            kernel_time = FindTune(h, solution);

                        json kern_objs = BuildJsonKernelList(h, solution.construction_params);

                        res_item["tunable"]   = solver.IsTunable();
                        res_item["params"]    = solver.GetPerfCfgParams(ctx, problem, db);
                        res_item["workspace"] = solution.workspace_sz;
                        res_item["time"]      = kernel_time;
                        res_item["layout"]    = problem.GetInLayout();
                        res_item["data_type"] = problem.GetXDesc().GetType();
                        res_item["direction"] = bn_dir;
                        // res_item["bias"]      = problem.GetBias(); dy?
                        res_item["kernel_objects"] = kern_objs;
                        res_item["reason"]         = "Success";
                        if(kernel_time == 0.0)
                            res_item["reason"] = "Invoker returned time = 0";
                        if(kernel_time < 0)
                            res_item["reson"] = "kernel_time not measured";
                    }
                    catch(const std::exception& e)
                    {
                        res_item["reason"] = std::string("Invoker exception: ") + e.what();
                        std::cerr << res_item["reason"] << std::endl;
                        return false;
                    }

                    return true;
                };

                auto res              = process_solver();
                res_item["evaluated"] = res;
                eval_result.push_back(res_item);
            }

        } // for each solver
    }     // for each solution

    if(tuning_op == TuningOp::Perf)
        output["miopen_perf_eval_result"] = eval_result;
    else if(tuning_op == TuningOp::Find)
        output["miopen_find_eval_result"] = eval_result;
    return 1;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
float BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::PerfTune(
    const miopen::Handle& h,
    const miopen::solver::ConvSolution& solution,
    miopen::PerformanceDb& db,
    miopen::ExecutionContext& perf_ctx)
{

    float kernel_time     = -1;
    perf_ctx.do_search    = true;
    perf_ctx.db_update    = true;
    const auto invoke_ctx = GetInvokeCtx();
    SolutionHasProgram(h, solution);

    const auto invoker = h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    kernel_time        = BaseFin::BenchmarkInvoker(invoker, h, invoke_ctx);

    return kernel_time;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
float BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::FindTune(
    const miopen::Handle& h, const miopen::solver::ConvSolution& solution)
{
    float kernel_time     = -1;
    const auto invoke_ctx = GetInvokeCtx();
    const auto invoker = h.PrepareInvoker(*solution.invoker_factory, solution.construction_params);
    kernel_time        = BaseFin::BenchmarkInvoker(invoker, h, invoke_ctx);
    return kernel_time;
}

template <typename TInput, typename Tref, typename TAcc, typename TScaleBias, typename TOut>
const miopen::AnyInvokeParams BNFin<TInput, Tref, TAcc, TScaleBias, TOut>::GetInvokeCtx()
{
    const auto invoke_ctx = [&] -> miopen::AnyInvokeParams {
        const auto bn_dir = GetDirection();
        if(bn_dir == miopen::debug::BatchNormDirection_t::ForwardTraining)
        {
            const auto ctx = [&]() {
                auto tmp                  = miopen::batchnorm::FwdTrainInvokeParams{};
                tmp.type                  = miopen::InvokeType::Run;
                tmp.x                     = in.GetVectorData();
                tmp.y                     = out.GetVectorData();
                tmp.bnScale               = bnScale.GetVectorData();
                tmp.bnBias                = bias.GetVectorData();
                tmp.expAvgFactor          = expAvgFactor;
                tmp.resultRunningMean     = runMean.GetVectorData();
                tmp.resultRunningVariance = runVariance.GetVectorData();
                tmp.epsilon               = epsilon;
                tmp.resultSaveMean        = savedMean.GetVectorData();
                tmp.resultSaveInvVariance = savedVariance.GetVectorData();
                return tmp;
            }();
            return ctx;
        }
        else if(bn_dir == miopen::debug::BatchNormDirection_t::ForwardInference)
        {
            const auto ctx = [&]() {
                auto tmp  = miopen::batchnorm::InfInvokeParams{};
                tmp.type  = miopen::InvokeType::Run;
                tmp.xDesc = &in.GetTensor().desc, tmp.x = in.GetVectorData();
                tmp.y                 = out.GetVectorData();
                tmp.bnScale           = bnScale.GetVectorData();
                tmp.bnBias            = bias.GetVectorData();
                tmp.estimatedMean     = estMean.GetVectorData();
                tmp.estimatedVariance = estVariance.GetVectorData();
                tmp.epsilon           = epsilon;
                return tmp;
            }();
            return ctx;
        }
        else if(bn_dir == miopen::debug::BatchNormDirection_t::Backward)
        {

            const auto ctx = [&]() {
                auto tmp    = miopen::batchnorm::BwdInvokeParams{};
                tmp.type    = miopen::InvokeType::Run;
                tmp.x       = in.GetVectorData();
                tmp.dy      = dy.GetVectorData();      //??right tensor?
                tmp.dx      = out_ref.GetVectorData(); //??right tensor?
                tmp.bnScale = bnScale.GetVectorData();
                // tmp.resultBnScaleDiff = resultBnScaleDiff; //??
                // tmp.resultBnBiasDiff  = resultBnBiasDiff; //??
                tmp.epsilon          = epsilon;
                tmp.savedMean        = savedMean.GetVectorData();
                tmp.savedInvVariance = savedVariance.GetVectorData();
                return tmp;
            }();
            return ctx;
        }
        else
        {
            std::ostringstream ss;
            ss << "Invalid Direction: " << static_cast<int>(bn_dir);
            throw std::runtime_error(ss.str());
        }
    };
    return invoke_ctx();
}

} // namespace fin
#endif // GUARD_MIOPEN_BN_FIN_HPP
