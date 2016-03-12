#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <string>

#include <deep_pyramid.h>
#include <fddb_container.h>

using namespace cv;
using namespace std;
using namespace caffe;

static const char argsDefs[] =
        "{ | config           |     | Path to configuration file }";

void printHelp(std::ostream& os)
{
    os << "\tUsage: --config=path/to/config.xml" << std::endl;
}

namespace ReturnCode
{
enum
{
    Success = 0,
    ConfigFileNotSpecified = 1,
    ConfigFileNotFound = 2,
    ImageFileNotFound = 3,
    TestFileNotFound = 4,
    OutputFileNotCreated = 5
};
};
int parseCommandLine(int argc, char *argv[], FileStorage& config)
{
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    if (configFileName.empty() == true)
    {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }

    config.open(configFileName, FileStorage::READ);

    if(config.isOpened()==false)
    {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    return ReturnCode::Success;
}

void readConfig(const FileStorage& config, string& model_file, string& trained_net_file,
                vector<string>& svm_file, vector<Size>& svmSize, int& levelCount, int& stride,
                string test_file, string image_folder, string output_file)
{
    config["NeuralNetwork-configuration"]>>model_file;
    config["NeuralNetwork-trained-model"]>>trained_net_file;
    config["NumberOfLevel"]>>levelCount;

    config["Stride"]>>stride;

    string svm_trained_file;
    config["SVM"]>>svm_trained_file;
    svm_file.push_back(svm_trained_file);
    Size filterSize;
    config["Filter-size"]>>filterSize;
    svmSize.push_back(filterSize);

    config["FileWithTestImage"]>>test_file;
    config["OutputFile"]>>output_file;
    config["TestImageFolder"]>>image_folder;
}

int main(int argc, char *argv[])
{
    FileStorage config;

    parseCommandLine(argc, argv, config);

    string model_file, trained_net_file;
    int levelCount;

    vector<string> svm_file;
    vector<Size> svmSize;
    int stride;

    string test_file, image_folder, output_file;

    readConfig(config, model_file, trained_net_file, svm_file, svmSize, levelCount, stride,
               test_file, image_folder, output_file);
    config.release();

    DeepPyramid pyramid(model_file, trained_net_file,svm_file, svmSize, levelCount, stride);
    FDDBContainer testData;
    testData.load(test_file, image_folder);

    FDDBContainer resultData;

    string img_path;
    for(int i=0;i<testData.size();i++)
    {
        Mat image;
        testData.next(image);

        vector<Rect> objects;
        vector<float> confidence;
        pyramid.detect(image, objects, confidence);

        resultData.add(image_folder+img_path+".jpg", objects, confidence);
    }

    resultData.save(output_file);

    return ReturnCode::Success;
}
