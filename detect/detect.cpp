#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <string.h>
#include <sys/stat.h>

#include <ie_version.hpp>
#include <ie_plugin_config.hpp>
#include <ie_plugin_dispatcher.hpp>
#include <ie_plugin_ptr.hpp>
#include <ie_extension.h>
#include <cpp/ie_cnn_net_reader.h>
#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <format_reader/format_reader_ptr.h>
#include <samples/common.hpp>

using namespace InferenceEngine::details;
using namespace InferenceEngine;

int main()
{
    // ---------------------Load plugin for inference engine------------------------------------------------
    std::string archPath = "../../lib/intel64";
    std::string FLAGS_pp = "";
    std::string FLAGS_d = "CPU";
    InferenceEngine::PluginDispatcher dispatcher({ FLAGS_pp, archPath , "" });
    InferenceEngine::InferenceEnginePluginPtr enginePtr;

    /** Loading plugin for device **/
    enginePtr = dispatcher.getPluginByDevice(FLAGS_d);

    /** Here we are loading the library with extensions if provided**/
    InferencePlugin plugin(enginePtr);

    /*If CPU device, load default library with extensions that comes with the product*/
    if (FLAGS_d.find("CPU") != std::string::npos) {
        /**
        * cpu_extensions library is compiled from "extension" folder containing
        * custom MKLDNNPlugin layer implementations. These layers are not supported
        * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    /** Setting plugin parameter for per layer metrics **/
    InferenceEngine::ResponseDesc resp;

    /** Printing plugin version **/
    const PluginVersion *pluginVersion;
    enginePtr->GetVersion((const InferenceEngine::Version*&)pluginVersion);
    std::cout << pluginVersion << std::endl << std::endl;

    // --------------------Load network (Generated xml/bin files)-------------------------------------------
    std::string networkName = "..\\model\\SSD_GoogleNetV2.xml";
    std::string binFileName = "..\\model\\SSD_GoogleNetV2.bin";

    InferenceEngine::CNNNetReader networkReader;
    /** Read network model **/
    networkReader.ReadNetwork(networkName);

    /** Extract model name and load weights **/
    networkReader.ReadWeights(binFileName);
    CNNNetwork network = networkReader.getNetwork();

    // -----------------------------Prepare input blobs-----------------------------------------------------
    /** Taking information about all topology inputs **/
    InferenceEngine::InputsDataMap inputsInfo(network.getInputsInfo());

    /** SSD network has one input and one output **/
    if (inputsInfo.size() != 1 && inputsInfo.size() != 2)
        return -1;

    /**
    * Some networks have SSD-like output format (ending with DetectionOutput layer), but
    * having 2 inputs as Faster-RCNN: one for image and one for "image info".
    *
    * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
    * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
    */
    std::string imageInputName, imInfoInputName;

    auto inputInfo = inputsInfo.begin()->second;

    InferenceEngine::SizeVector inputImageDims;
    /** Stores input image **/

    /** Iterating over all input blobs **/
    for (auto & item : inputsInfo) {
        /** Working with first input tensor that stores image **/
        if (item.second->getInputData()->dims.size() == 4) {
            imageInputName = item.first;

            /** Creating first input blob **/
            Precision inputPrecision = Precision::U8;
            item.second->setPrecision(inputPrecision);
        }
        else if (item.second->getInputData()->dims.size() == 2) {
            imInfoInputName = item.first;

            Precision inputPrecision = Precision::FP32;
            item.second->setPrecision(inputPrecision);
            if ((item.second->getDims()[0] != 3 && item.second->getDims()[0] != 6) || item.second->getDims()[1] != 1) {
                return -1;
            }
        }
    }

    // ---------------------------Prepare output blobs------------------------------------------------------
    InferenceEngine::OutputsDataMap outputsInfo(network.getOutputsInfo());

    std::string outputName;
    DataPtr outputInfo;
    for (const auto& out : outputsInfo) {
        if (out.second->creatorLayer.lock()->type == "DetectionOutput") {
            outputName = out.first;
            outputInfo = out.second;
        }
    }

    if (outputInfo == nullptr) {
        return -1;
    }

    const InferenceEngine::SizeVector outputDims = outputInfo->dims;

    const int maxProposalCount = outputDims[1];
    const int objectSize = outputDims[0];

    if (objectSize != 7) {
        return -1;
    }

    if (outputDims.size() != 4) {
        return -1;
    }

    /** Set the precision of output data provided by the user, should be called before load of the network to the plugin **/
    outputInfo->setPrecision(Precision::FP32);

    // -------------------------Loading model to the plugin-------------------------------------------------
    auto executable_network = plugin.LoadNetwork(network, {});
    auto infer_request = executable_network.CreateInferRequest();


    // -------------------------------Set input data--------------------------------------------------------
    /** Collect images data ptrs **/
    std::vector<std::string> images = {"cat3.jpg"};
    std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
    std::vector<int> imageWidths, imageHeights;
    for (auto & i : images) {
        FormatReader::ReaderPtr reader(i.c_str());
        if (reader.get() == nullptr) {
            continue;
        }
        /** Store image data **/
        std::shared_ptr<unsigned char> originalData(reader->getData());
        std::shared_ptr<unsigned char> data(reader->getData(inputInfo->getDims()[0], inputInfo->getDims()[1]));
        if (data.get() != nullptr) {
            originalImagesData.push_back(originalData);
            imagesData.push_back(data);
            imageWidths.push_back(reader->width());
            imageHeights.push_back(reader->height());
        }
    }
    if (imagesData.empty()) 
        return -1;

    size_t batchSize = network.getBatchSize();
    if (batchSize != imagesData.size()) {
        batchSize = std::min(batchSize, imagesData.size());
    }

    /** Creating input blob **/
    Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

    /** Filling input tensor with images. First b channel, then g and r channels **/
    size_t num_channels = imageInput->dims()[2];
    size_t image_size = imageInput->dims()[1] * imageInput->dims()[0];

    unsigned char* data = static_cast<unsigned char*>(imageInput->buffer());

    /** Iterate over all input images **/
    for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (size_t pid = 0; pid < image_size; pid++) {
            /** Iterate over all channels **/
            for (size_t ch = 0; ch < num_channels; ++ch) {
                /**          [images stride + channels stride + pixel id ] all in bytes            **/
                data[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
            }
        }
    }

    if (imInfoInputName != "") {
        auto input2 = infer_request.GetBlob(imInfoInputName);
        auto imInfoDim = inputsInfo.find(imInfoInputName)->second->getDims()[0];

        /** Fill input tensor with values **/
        float *p = input2->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

        for (size_t image_id = 0; image_id < std::min(imagesData.size(), batchSize); ++image_id) {
            p[image_id * imInfoDim + 0] = static_cast<float>(inputsInfo[imageInputName]->getDims()[1]);
            p[image_id * imInfoDim + 1] = static_cast<float>(inputsInfo[imageInputName]->getDims()[0]);
            for (int k = 2; k < imInfoDim; k++) {
                p[image_id * imInfoDim + k] = 1.0f;  // all scale factors are set to 1.0
            }
        }
    }

    // ----------------------------Do inference-------------------------------------------------------------
    int FLAGS_ni = 1;
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
    typedef std::chrono::duration<float> fsec;

    double total = 0.0;
    /** Start inference & calc performance **/
    for (int iter = 0; iter < FLAGS_ni; ++iter) {
        auto t0 = Time::now();
        infer_request.Infer();
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        ms d = std::chrono::duration_cast<ms>(fs);
        total += d.count();
    }

    // ---------------------------Post-process output blobs--------------------------------------------------
    const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
    const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

    std::vector<std::vector<int> > boxes(batchSize);
    std::vector<std::vector<int> > classes(batchSize);

    /* Each detection has image_id that denotes processed image */
    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
        float image_id = detection[curProposal * objectSize + 0];
        float label = detection[curProposal * objectSize + 1];
        float confidence = detection[curProposal * objectSize + 2];
        float xmin = detection[curProposal * objectSize + 3] * imageWidths[image_id];
        float ymin = detection[curProposal * objectSize + 4] * imageHeights[image_id];
        float xmax = detection[curProposal * objectSize + 5] * imageWidths[image_id];
        float ymax = detection[curProposal * objectSize + 6] * imageHeights[image_id];

        /* MKLDnn and clDNN have little difference in DetectionOutput layer, so we need this check */
        if (image_id < 0 || confidence == 0) {
            continue;
        }

        std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
            "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id;

        if (confidence > 0.5) {
            /** Drawing only objects with >50% probability **/
            classes[image_id].push_back(static_cast<int>(label));
            boxes[image_id].push_back(static_cast<int>(xmin));
            boxes[image_id].push_back(static_cast<int>(ymin));
            boxes[image_id].push_back(static_cast<int>(xmax - xmin));
            boxes[image_id].push_back(static_cast<int>(ymax - ymin));
            std::cout << " WILL BE PRINTED!";
        }
        std::cout << std::endl;
    }

    for (size_t batch_id = 0; batch_id < batchSize; ++batch_id) {
        addRectangles(originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id], boxes[batch_id], classes[batch_id]);
        const std::string image_path = "out_" + std::to_string(batch_id) + ".bmp";
        if (writeOutputBmp(image_path, originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id])) {
            std::cout << "Image " + image_path + " created!" << std::endl;
        }
        else {
            return -1;
        }
    }

    return 0;
}
