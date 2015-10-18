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
    Caffe::set_mode(Caffe::CPU);
    Mat image;

    string alexnet_model_file=argv[1];
    string alexnet_trained_file=argv[2];

    DeepPyramid pyramid(7,alexnet_model_file, alexnet_trained_file);
    string image_file=argv[3];

    image=imread(image_file, CV_LOAD_IMAGE_COLOR);
    pyramid.detect(image);
    //namedWindow("Image", WINDOW_AUTOSIZE);
    //imshow("Image",image);
    //waitKey(0);
    //pyramid.calculate(image);
    return 0;
}
