#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <deep_pyramid.h>
#include <fstream>
#include <string>
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

class TestConfiguration
{
public:
    string test_file_path;

    string output_file_path;

    string test_image_folder;

    string result_image_folder;

    bool saveImage;

    TestConfiguration(FileStorage& config)
    {
        config["FileWithTestImage"]>>test_file_path;
        config["OutputFile"]>>output_file_path;
        config["TestImageFolder"]>>test_image_folder;
        config["TestImageResultFolder"]>>result_image_folder;
        config["SaveTestImageResult"]>>saveImage;
    }

};

void writeDetectionResultInFile(ofstream& file, const string& img_path, const vector<ObjectBox> & objects)
{
    file<<img_path<<endl;
    file<<objects.size()<<endl;
    for(unsigned int i=0;i<objects.size();i++)
    {
        file<<objects[i].originalImageBox.x<<" ";
        file<<objects[i].originalImageBox.y<<" ";
        file<<objects[i].originalImageBox.width<<" ";
        file<<objects[i].originalImageBox.height<<" ";
        file<<objects[i].confidence<<endl;
    }
}

void drawObjects(const Mat& src, Mat& dst, const vector<ObjectBox>& objects)
{
    src.copyTo(dst);
    for(unsigned int i=0; i<objects.size();i++)
    {
        rectangle(dst, objects[i].originalImageBox, Scalar(0,255,0));
    }
}

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");

    if (configFileName.empty() == true)
    {
        std::cerr << "Configuration file is not specified." << std::endl;
        printHelp(std::cerr);
        return ReturnCode::ConfigFileNotSpecified;
    }

    FileStorage config(configFileName, FileStorage::READ);

    if(config.isOpened()==false)
    {
        std::cerr << "File '" << configFileName
                  << "' not found. Exiting." << std::endl;
        return ReturnCode::ConfigFileNotFound;
    }

    TestConfiguration testConfig(config);

    DeepPyramid pyramid(config);

    ifstream test_file(testConfig.test_file_path);

    if(test_file.is_open()==false)
    {
        std::cerr << "Test file '" << testConfig.test_file_path
                  << "' not found. Exiting" << std::endl;
        return ReturnCode::TestFileNotFound;
    }

    ofstream output_file(testConfig.output_file_path);

    if(output_file.is_open()==false)
    {
        std::cerr << "Output file '" << testConfig.output_file_path
                  << "' not created. Exiting" << std::endl;
        return ReturnCode::OutputFileNotCreated;
    }

    string img_path;
    while(test_file>>img_path)
    {
        Mat image;
        image=imread(testConfig.test_image_folder+img_path+".jpg");

        if(!image.data)
        {
            std::cerr << "File '" << testConfig.test_image_folder+img_path+".jpg"
                      << "' not found. Exiting." << std::endl;
            return ReturnCode::ImageFileNotFound;
        }

        vector<ObjectBox> objects;
        pyramid.detect(image, objects);

        writeDetectionResultInFile(output_file, img_path, objects);

        if(testConfig.saveImage)
        {
            Mat imageWithObjects;
            drawObjects(image, imageWithObjects, objects);

            imwrite(testConfig.result_image_folder+img_path+".jpg", imageWithObjects);

            std::replace( img_path.begin(), img_path.end(), '/', '_');
            cout<<"SAVE:"<<testConfig.result_image_folder+img_path+".jpg"<<endl;
        }
    }
    test_file.close();
    output_file.close();

    return ReturnCode::Success;
}
