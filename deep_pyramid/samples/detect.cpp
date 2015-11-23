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

    DeepPyramid pyramid(argv[1]);
    string image_file=argv[2];

    image=imread(image_file);

    pyramid.detect(image);

    imwrite(image_file+"_res.jpg", pyramid.getImageWithObjects());
    cout<<"here!"<<endl;
    return 0;
}
