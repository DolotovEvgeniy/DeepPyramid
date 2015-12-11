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

int main(int argc, char *argv[])
{
    string config_file=argv[1];
    DeepPyramid pyramid(config_file);

    Mat image;
    string image_file=argv[2];
    image=imread(image_file);

    vector<ObjectBox> objects;
    pyramid.detect(image, objects);

    Mat imageWithObjects;
    image.copyTo(imageWithObjects);
    for(unsigned int i=0; i<objects.size();i++)
    {
        rectangle(imageWithObjects, objects[i].originalImageBox, Scalar(0,255,0));
    }
    imwrite(image_file+"_res.jpg", imageWithObjects);

    return 0;
}
