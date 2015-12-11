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

int main(int argc, char *argv[])
{
    cv::CommandLineParser parser(argc, argv, argsDefs);
    string configFileName = parser.get<std::string>("config");
    DeepPyramid pyramid(configFileName);

    Mat image;
    string imageFileName = parser.get<std::string>("image");
    image=imread(imageFileName);

    vector<ObjectBox> objects;
    pyramid.detect(image, objects);

    Mat imageWithObjects;
    image.copyTo(imageWithObjects);
    for(unsigned int i=0; i<objects.size();i++)
    {
        rectangle(imageWithObjects, objects[i].originalImageBox, Scalar(0,255,0));
    }
    imwrite(imageFileName+"_res.jpg", imageWithObjects);

    return 0;
}
