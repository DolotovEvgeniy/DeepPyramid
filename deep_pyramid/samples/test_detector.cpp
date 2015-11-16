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

int main(int argc, char *argv[])
{
    Caffe::set_mode(Caffe::CPU);

    string alexnet_model_file=argv[1];
    string alexnet_trained_file=argv[2];
    string svm_trained_file=argv[3];
    DeepPyramid pyramid(7,alexnet_model_file, alexnet_trained_file);

    CvSVM classifier;
    classifier.load(svm_trained_file.c_str());
    pyramid.addRootFilter(Size(atoi(argv[4]),atoi(argv[5])),&classifier);
    string test_file=argv[6];
    ifstream input_file(test_file.c_str());
    ofstream output_file(argv[7]);
    string img_path;
    while(input_file>>img_path)
    {
        Mat image;
        image=imread(img_path+".jpg");
        vector<ObjectBox> objects=pyramid.detect(image);
        output_file<<img_path<<endl;
        output_file<<objects.size()<<endl;
        for(unsigned int i=0;i<objects.size();i++)
        {
            output_file<<objects[i].originalImageBox.x<<" ";
            output_file<<objects[i].originalImageBox.y<<" ";
            output_file<<objects[i].originalImageBox.width<<" ";
            output_file<<objects[i].originalImageBox.height<<" ";
            output_file<<objects[i].confidence<<endl;
        }

    }
    input_file.close();
    output_file.close();

    return 0;
}
