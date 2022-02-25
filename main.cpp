/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                                  \
    if (!(x))                                                    \
    {                                                            \
        fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
        exit(1);                                                 \
    }

int main(int argc, char *argv[])
{
    // cv::Mat image = cv::imread("//Users//aparajitgarg//Desktop//IMG_20190129_110614_Bokeh.jpg");
    // std::cout << "=================================================================================" << std::endl;
    // std::cout << image.rows << " " << image.cols << std::endl;
    // std::cout << "=================================================================================";
    if (argc != 2)
    {
        fprintf(stderr, "minimal <tflite model>\n");
        return 1;
    }
    const char *filename = argv[1];
    const std::string image_path = "1.png"; //argv[2];
    // const string image_path = '1.png'

    std::cout << "Filename: " << *filename << std::endl;
    // std::cout << "Image Name: " << *image_path << std::endl;

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
    
    if (model == nullptr){
        std::cout << "Failed to load model";
        return 0;
    }

    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Intrepter and does various set up
    // tasks so that the Interpreter can read the provided model.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);

    if (interpreter == nullptr){
        std::cout << "Failed to initialize the interpreter";
        return 0;
    }

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    printf("=== Pre-invoke Interpreter State ===\n");
    tflite::PrintInterpreterState(interpreter.get());

    // Fill input buffers
    // TODO(user): Insert code to fill input tensors.
    // Note: The buffer of the input tensor with index `i` of type T can
    // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

    interpreter->SetNumThreads(1);
    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];

    //Load input image
    cv::Mat image;
    auto frame = cv::imread(image_path);

    if(frame.empty()){
        std::cout << "Failed to load the image" << std::endl;
        return 0;
    }
    std::cout << "++++++image read successfully++++++++" << std::endl;

    // Copy image to input tensor
    cv::resize(frame, image, cv::Size(width, height), cv::INTER_NEAREST);
    // uchar *input = interpreter->typed_input_tensor<uchar>(0);

    std::cout << "+++++++++RESIZE COMPLETED++++++++++++++" << image.rows << " " << image.cols << " " << std::endl;
    memcpy(interpreter->typed_tensor<float>(0), image.data, image.total() * image.elemSize());

    std::cout << "++++++++++COPY FUNCTION EXECUTED SUCCESSFULLY++++++++++++++" << std::endl;

    std::cout << "tensors size: " << interpreter->tensors_size() << "\n";
    std::cout << "nodes size: " << interpreter->nodes_size() << "\n";
    std::cout << "inputs: " << interpreter->inputs().size() << "\n";
    std::cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
    int image_width = 160;
    int image_height = 160;
    int image_channels = 3;
    std::vector<uint8_t> in = read_bmp(s->input_bmp_name, &image_width,
                                       &image_height, &image_channels, 1);
    //Inference
    std::chrono::steady_clock::time_point start, end;
    start = std::chrono::steady_clock::now();
    interpreter->Invoke();
    end = std::chrono::steady_clock::now();

    std::cout << "+++++++++++INVOKED PROPERLY AND END EXECUTED++++++++" << std::endl;
    auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds> (end - start).count();

    // float *output = interpreter->typed_output_tensor<float>(0);
    // std::cout << "OUTPUT: " << output[0] << std::endl;
    int output = interpreter->outputs()[0];
    for (auto i = 0; i < interpreter->outputs().size(); i++) {
        std::cout << "i value: " << interpreter->outputs()[i] << std::endl;
    }
         //interpreter->outputs()[i];
    // cout << interpreter->outputs();
    TfLiteIntArray *output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    std::cout << "OUTPUT DIMS: " << output_dims << std::endl;
    std::cout << "OUTPUT SIZE: " << output_size << std::endl;
    std::cout << output << " " << output_size << std::endl;
    std::cout << (interpreter->tensor(output)->type) << std::endl;


    
    // Read output buffers
    // TODO(user): Insert getting data out code.
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

        return 0;
}
