#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <array>
#include <memory>
#include <json.hpp>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

using namespace std;
using namespace std::chrono;
using json = nlohmann::json;

/* ====================== 原有 Scaler，不改 ====================== */

struct Scaler {
    std::vector<float> mean;
    std::vector<float> scale;

    std::vector<float> transform(const std::vector<float>& x) const {
        std::vector<float> result(x.size());
        for (size_t i = 0; i < x.size(); ++i)
            result[i] = (x[i] - mean[i]) / scale[i];
        return result;
    }
};

Scaler load_scaler(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open())
        throw std::runtime_error("Failed to open scaler JSON");

    json j;
    f >> j;

    Scaler s;
    s.mean = j["mean"].get<std::vector<float>>();
    s.scale = j["scale"].get<std::vector<float>>();
    return s;
}

std::pair<float, float> load_scale_node(
    const std::string& path, int outputdim
) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Failed to open scale.json");

    json data;
    f >> data;

    return {
        data["scale_node"].get<float>(),
        data["scale_Dis"][outputdim].get<float>()
    };
}

/* ====================== 新增：推理上下文 ====================== */

struct InferenceContext {
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::vector<std::unique_ptr<Ort::Session>> sessions;

    Scaler scaler;
    float scale_node;
    std::vector<float> scale_S;
};

/* ====================== 初始化（只调用一次） ====================== */

InferenceContext init_context(
    const std::string& folder_name,
    bool use_standerscale,
    bool use_gpu
) {
    InferenceContext ctx{
        Ort::Env(ORT_LOGGING_LEVEL_WARNING, "inference"),
        Ort::SessionOptions()
    };

    ctx.session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL
    );
    ctx.session_options.SetIntraOpNumThreads(8);

    if (use_gpu) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;

        api->CreateCUDAProviderOptions(&cuda_options);

        OrtStatus* status = nullptr;
        status = api->SessionOptionsAppendExecutionProvider_CUDA_V2(
            ctx.session_options, cuda_options
        );
        api->ReleaseCUDAProviderOptions(cuda_options);

        if (status != nullptr) {
            api->ReleaseStatus(status);
            std::cout << "Falling back to CPU Execution Provider\n";
        } else {
            std::cout << "Using CUDA Execution Provider (GPU)\n";
        }
    } else {
        std::cout << "Using CPU Execution Provider\n";
    }

    if (use_standerscale) {
        ctx.scaler = load_scaler(
            folder_name + "/dataset/scaler_inputs.json"
        );
    }

    ctx.scale_S.resize(9);
    auto scale_json =
        json::parse(std::ifstream(folder_name + "/dataset/scale.json"));

    ctx.scale_node = scale_json["scale_node"];
    for (int i = 0; i < 9; ++i)
        ctx.scale_S[i] = scale_json["scale_Dis"][i];

    for (int i = 0; i < 9; ++i) {
        std::string model =
            folder_name + "/models/Dis_" + std::to_string(i) + ".onnx";
        std::wstring w(model.begin(), model.end());

        ctx.sessions.emplace_back(
            std::make_unique<Ort::Session>(
                ctx.env, w.c_str(), ctx.session_options
            )
        );
        std::cout << "Model loaded: " << model << std::endl;
    }

    return ctx;
}

/* ====================== 纯 Run，不做初始化 ====================== */

std::vector<float> run_inference(
    Ort::Session& session,
    const std::vector<std::vector<float>>& nodepairs_vec,
    const InferenceContext& ctx,
    int outputdim,
    bool use_standerscale
) {
    size_t batch = nodepairs_vec.size();
    size_t dim = nodepairs_vec[0].size();

    std::vector<float> input(batch * dim);

    for (size_t i = 0; i < batch; ++i) {
        std::vector<float> tmp = nodepairs_vec[i];
        for (auto& v : tmp) v *= ctx.scale_node;
        if (use_standerscale)
            tmp = ctx.scaler.transform(tmp);

        std::copy(
            tmp.begin(), tmp.end(),
            input.begin() + i * dim
        );
    }

    std::array<int64_t, 2> shape{
        static_cast<int64_t>(batch),
        static_cast<int64_t>(dim)
    };

    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemTypeCPU
        );

    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            mem,
            input.data(),
            input.size(),
            shape.data(),
            2
        );

    auto allocator = Ort::AllocatorWithDefaultOptions();
    auto in_name_alloc  = session.GetInputNameAllocated(0, allocator);
    auto out_name_alloc = session.GetOutputNameAllocated(0, allocator);

    const char* input_names[]  = { in_name_alloc.get() };
    const char* output_names[] = { out_name_alloc.get() };

    auto start = high_resolution_clock::now();
    auto outputs = session.Run(
        Ort::RunOptions{ nullptr },
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );
    auto end = high_resolution_clock::now();

   /* std::cout << "Run time: "
              << duration_cast<duration<double>>(end - start).count()
              << " sec\n";*/

    float* ptr = outputs[0].GetTensorMutableData<float>();
    std::vector<float> result(ptr, ptr + batch);

    for (auto& v : result)
        v /= ctx.scale_S[outputdim];

    return result;
}

/* ====================== main（结构保持） ====================== */

int main() {
    std::cout << "ORT Runtime Version: "
              << Ort::GetVersionString() << std::endl;

    try {
        
        std::string folder_name = "D:/inference_cpp/inference";
        bool use_standerscale = true;
        bool use_gpu = true;
        const size_t BATCH_SIZE = 412;
        int iter_num=500;

        std::vector<std::vector<float>> base_nodepairs = {
            { -0.0189422593f, 0.00459089829f, 0.00200000009f, -0.0194065776f,
            -0.0018083906f, 0.00600000005f, 0.00046432f, 0.00639929f, -0.004f },
            { 0.0101501f, 0.01553772f, 0.f, 0.004f, 0.f, 0.f,
            0.0061501f, 0.01553772f, 0.0f },
            { 0.0101501f, 0.01553772f, 0.f, 0.00354182f, 0.00185889f,
            0.01f, 0.00660827f, 0.01367883f, -0.01f }
        };

        std::vector<std::vector<float>> nodepairs;
        nodepairs.reserve(BATCH_SIZE);

        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            nodepairs.push_back(base_nodepairs[i % base_nodepairs.size()]);
        }

        cout << "input size " << nodepairs.size() << endl;

        auto s0 = high_resolution_clock::now();
        auto ctx = init_context(
            folder_name,
            use_standerscale,
            use_gpu
        );
        auto s1 = high_resolution_clock::now();

        std::vector<std::vector<float>> all_results(
            nodepairs.size(),
            std::vector<float>(9)
        );

        for (int iter = 0; iter < iter_num; ++iter)
        {
            std::cout <<"iter" << iter << std::endl;
            auto t0 = high_resolution_clock::now();
            for (int i = 0; i < 9; ++i) {
                for(int batchId = 0; batchId < 62; ++batchId){
                    auto res = run_inference(
                        *ctx.sessions[i],
                        nodepairs,
                        ctx,
                        i,
                        use_standerscale
                    );

                    for (size_t j = 0; j < res.size(); ++j)
                        all_results[j][i] = res[j];
                }
            }
            auto t1 = high_resolution_clock::now();

            std::cout << "iter time: "
                  << duration_cast<duration<double>>(t1 - t0).count()
                  << " sec\n";
        }
        auto r1 = high_resolution_clock::now();

        std::cout << "initial time: "
            << duration_cast<duration<double>>(s1 - s0).count()
            << " sec\n";

        std::cout << "inference time: "
            << duration_cast<duration<double>>(r1 - s1).count()
            << " sec\n";

        std::cout << "total time: "
                  << duration_cast<duration<double>>(r1 - s0).count()
                  << " sec\n";
        
        std::cout << "iter time aver: "
                  << duration_cast<duration<double>>(r1 - s1).count()/iter_num
                  << " sec\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
