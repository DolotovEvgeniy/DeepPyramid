#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <deep_pyramid.h>

using namespace cv;
using namespace std;
using namespace caffe;

static const char argsDefs[] =
        "{ | config           |     | Path to configuration file }"
        "{ | image            |     | Path to image              }";

void printHelp(std::ostream& os)
{
    os << "\tUsage: --config=path/to/config.xml --image=input/image/filename" << std::endl;
}

namespace ReturnCode
{
enum
{
    Success = 0,
    ConfigFileNotSpecified = 1,
    ImageFileNotSpecified = 2,
    ConfigFileNotFound = 3,
    ImageFileNotFound = 4
};
};

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    Mat image;
    string imageFileName = parser.get<std::string>("image");


    if (configFileName.empty() == true)
    {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }
    if (imageFileName.empty() == true)
    {
        std::cerr << "Image file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ImageFileNotSpecified;
    }

    FileStorage config(configFileName, FileStorage::READ);

    if(config.isOpened()==false)
    {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    image=imread(imageFileName);

    if(!image.data)
    {
        std::cerr << "File '" << imageFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ImageFileNotFound;
    }

    DeepPyramid pyramid(config);

    vector<ObjectBox> objects;
    pyramid.detect(image, objects);

    Mat imageWithObjects;
    image.copyTo(imageWithObjects);
    for(unsigned int i=0; i<objects.size();i++)
    {
        rectangle(imageWithObjects, objects[i].originalImageBox, Scalar(0,255,0));
    }
    imwrite(imageFileName+"_res.jpg", imageWithObjects);

    config.release();

    return ReturnCode::Success;
}
