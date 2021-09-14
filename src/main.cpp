/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#include <miopen/miopen.h>

#include "conv_fin.hpp"
#include "error.hpp"
#include "fin.hpp"

#include <half.hpp>
#include <miopen/bfloat16.hpp>

using half_float::half;
typedef half float16;

#include <miopen/tensor.hpp>
#include <nlohmann/json.hpp>
#include <typeinfo>

#include <algorithm>
#include <cstdio>
#include <half.hpp>
#include <iostream>

using json = nlohmann::json;

[[gnu::noreturn]] void Usage()
{
    printf("Usage: ./fin *input_json *output_json\n\n");
    printf("Supported arguments:\n");
    printf("-i *input_json\n");
    printf("-o *output_json\n");
    printf("\n");
    exit(0);
}

int main(int argc, char* argv[], char* envp[])
{
    std::vector<std::string> args(argv, argv + argc);
    std::string ifile;
    std::string ofile;
    std::map<char, std::string> MapInputs = {};

    for(auto& arg : args)
    {
        if(arg == "--help" || arg == "-help" || arg == "-h")
        {
            Usage();
        }
    }

    if(argc != 5)
    {
        std::cerr << "Invalid arguments" << std::endl;
        Usage();
    }

    for(int i = 0; i < args.size(); i++)
    {
        if(args[i] == "-i")
        {
            if(!boost::filesystem::exists(args[i + 1]))
            {
                std::cerr << "File: " << args[i + 1] << " does not exist" << std::endl;
                exit(-1);
            }
            MapInputs[args[i].back()] = args[i + 1];
        }
        if(args[i] == "-o")
        {
            ofile                     = args[i + 1];
            MapInputs[args[i].back()] = args[i + 1];
        }
    }

    boost::filesystem::path input_filename(MapInputs['i']);
    boost::filesystem::path output_filename(MapInputs['o']);

    // The JSON is a list of commands, so we iterate over the list and then
    // process each map
    std::ifstream i(input_filename.string());
    // TODO: fix the output writing so that interim results are not lost if one of
    // the iterations crash
    std::ofstream o(output_filename.string());
    json j; //  = json::parse(cmd);
    i >> j;
    i.close();
    json final_output;
    // Get the process env
    std::vector<std::string> jenv;
    for(auto env = envp; *env != nullptr; env++)
        jenv.push_back(*env);
    json res_item;

    res_item["process_env"] = jenv;
    final_output.push_back(res_item);
    // process through the jobs
    for(auto& it : j)
    {
        auto command = it;
        fin::Fin* f  = nullptr;
        // TODO : Move this to a factory function
        if(command.contains("config"))
        {
            if(command["config"]["cmd"] == "conv")
            {
                f = new fin::ConvFin<float, float>(command);
            }
            else if(command["config"]["cmd"] == "convfp16")
            {
                f = new fin::ConvFin<float16, float>(command);
            }
            else if(command["config"]["cmd"] == "convbfp16")
            {
                f = new fin::ConvFin<bfloat16, float>(command);
            }
            else
            {
                FIN_THROW("Invalid operation: " + command["config"]["cmd"].get<std::string>());
                exit(-1);
            }
        }
        else
        {
            f = new fin::ConvFin<float, float>();
            dynamic_cast<fin::ConvFin<float,float>*>(f)->job["arch"] = command["arch"];
            dynamic_cast<fin::ConvFin<float,float>*>(f)->job["num_cu"] = command["num_cu"];
        }

        for(auto& step_it : command["steps"])
        {
            std::string step = step_it.get<std::string>();
            f->ProcessStep(step);
        }
        f->output["config_tuna_id"] = command["config_tuna_id"];
        f->output["arch"]           = command["arch"];
        f->output["direction"]      = command["direction"];
        f->output["input"]          = command;
        final_output.push_back(f->output);
    }
    o << std::setw(4) << final_output << std::endl;
    o.flush();
    o.close();
    return 0;
}
