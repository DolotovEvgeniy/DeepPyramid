// Copyright 2016 Dolotov Evgeniy

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <deep_pyramid.h>
#include <fddb_container.h>
#include <detect_result_container.h>

using namespace cv;
using namespace std;
using namespace caffe;

static const char argsDefs[] =
        "{ | config           |     | Path to configuration file }";

void printHelp(std::ostream& os) {
    os << "\tUsage: --config=path/to/config.xml" << std::endl;
}

namespace ReturnCode {
enum {
    Success = 0,
    ConfigFileNotSpecified = 1,
    ConfigFileNotFound = 2,
    ImageFileNotFound = 3,
    TestFileNotFound = 4,
    OutputFileNotCreated = 5
};
};
int parseCommandLine(int argc, char *argv[], FileStorage& config) {
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    if (configFileName.empty() == true) {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }

    config.open(configFileName, FileStorage::READ);

    if (config.isOpened() == false) {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    return ReturnCode::Success;
}

int main(int argc, char *argv[]) {
    FileStorage config;

    parseCommandLine(argc, argv, config);

    DeepPyramid pyramid(config);

    string test_data_filename;
    string test_data_folder;
    config["test_data"] >> test_data_filename;
    config["test_data_folder"] >> test_data_folder;

    FDDBContainer testData;
    testData.load(test_data_filename, test_data_folder);
    cout << testData.size() <<endl;
    DetectResultContainer resultData;

    for(int i = 0; i < testData.size(); i++) {
        cout << "Processing " << i+1 <<" image. "
             << testData.size()-i-1 << " images have not processed yet!" <<endl;
        string img_path;
        Mat image;
        vector<Rect> objects;
        testData.next(img_path, image, objects);

        vector<Rect> detectedObjects;
        vector<float> confidence;
        pyramid.detect(image, detectedObjects, confidence);

        resultData.add(img_path, detectedObjects, confidence);
    }

    string output_filename;
    config["result_file"] >> output_filename;
    resultData.save(output_filename);

    config.release();

    return ReturnCode::Success;
}
