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
int parseCommandLine(int argc, char *argv[], string& config_file)
{
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    if (configFileName.empty() == true)
    {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }

    config_file=configFileName;

    return ReturnCode::Success;
}

void readConfig(const FileStorage& config, string& model_file, string& trained_net_file,
                vector<string>& svm_file, vector<Size>& svmSize,string& box_regressor_file, int& levelCount, int& stride,
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
    string config;

    parseCommandLine(argc, argv, config);

    string test_file, image_folder, output_file;

    DeepPyramid pyramid(config);

    FDDBContainer testData;
    testData.load(test_file, image_folder);

    FDDBContainer resultData;

    for(int i=0;i<testData.size();i++)
    {
        Mat image;
        string img_path;

        testData.next(image, img_path);

        vector<BoundingBox> objects;
        pyramid.detect(image, objects);

        vector<Rect> rectangles;
        vector<float> confidence;
        for(unsigned int i=0;i<objects.size();i++)
        {
            rectangles.push_back(objects[i].originalImageBox);

            confidence.push_back(objects[i].confidence);
        }
        resultData.add(image_folder+img_path+".jpg", rectangles, confidence);
    }

    resultData.save(output_file);

    return ReturnCode::Success;
}
