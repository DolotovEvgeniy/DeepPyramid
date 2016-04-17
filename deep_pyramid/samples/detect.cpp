// Copyright 2016 Dolotov Evgeniy

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>

#include <deep_pyramid.h>

using namespace cv;
using namespace std;
using namespace caffe;

static const char argsDefs[] =
        "{ | config           |     | Path to configuration file }"
        "{ | image            |     | Path to image              }";

void printHelp(std::ostream& os) {
    os << "\tUsage: --config=path/to/config.xml --image=input/image/filename" << std::endl;
}

namespace ReturnCode {
enum {
    Success = 0,
    ConfigFileNotSpecified = 1,
    ImageFileNotSpecified = 2,
    ConfigFileNotFound = 3,
    ImageFileNotFound = 4
};
};

int parseCommandLine(int argc, char *argv[], Mat& image, FileStorage& config) {
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");
    string imageFileName = parser.get<std::string>("image");

    if (configFileName.empty() == true) {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }
    if (imageFileName.empty() == true) {
        std::cerr << "Image file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ImageFileNotSpecified;
    }

    config.open(configFileName, FileStorage::READ);

    if (config.isOpened() == false) {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    image = imread(imageFileName);

    if (!image.data) {
        std::cerr << "File '" << imageFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ImageFileNotFound;
    }

    return ReturnCode::Success;
}

void saveImageWithObjects(string file_name, const Mat& image, const vector<Rect>& objects) {
    Mat imageWithObjects;
    image.copyTo(imageWithObjects);
    for (unsigned int i = 0; i < objects.size(); i++) {
        rectangle(imageWithObjects, objects[i], Scalar(0, 255, 0));
    }
    imwrite(file_name, imageWithObjects);
}

int main(int argc, char *argv[]) {
    Mat image;
    FileStorage config;

    int returnCode = parseCommandLine(argc, argv, image, config);

    if (returnCode != ReturnCode::Success) {
        return returnCode;
    }

    DeepPyramid pyramid(config);
    vector<Rect> objects;
    vector<float> confidence;
    pyramid.detect(image, objects, confidence);

    saveImageWithObjects("detectedObjects.jpg", image, objects);

    return ReturnCode::Success;
}
