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
    string svm_trained_file=argv[3];
    DeepPyramid pyramid(7,alexnet_model_file, alexnet_trained_file);
    pyramid.load("pca.xml");
    string image_file=argv[4];

    CvSVM classifier;
    classifier.load(svm_trained_file.c_str());
    pyramid.addRootFilter(Size(7,11),&classifier);

    image=imread(image_file, CV_LOAD_IMAGE_COLOR);

    pyramid.detect(image);

    return 0;
}
