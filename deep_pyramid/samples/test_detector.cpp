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
    string config_file=argv[1];
    DeepPyramid pyramid(config_file, DeepPyramidMode::TEST);

    FileStorage config(config_file, FileStorage::READ);

    string test_file_path;
    config["FileWithTestImage"]>>test_file_path;

    string output_file_path;
    config["OutputFile"]>>output_file_path;

    string test_image_folder;
    config["TestImageFolder"]>>test_image_folder;

    string result_image_folder;
    config["TestImageResultFolder"]>>result_image_folder;

    bool saveImage;
    config["SaveTestImageResult"]>>saveImage;

    ifstream test_file(test_file_path);
    ofstream output_file(output_file_path);

    string img_path;
    while(test_file>>img_path)
    {
        Mat image;
        image=imread(test_image_folder+img_path+".jpg");
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

        if(saveImage)
        {
            string test_image_name=img_path;
            std::replace( test_image_name.begin(), test_image_name.end(), '/', '_');
            cout<<"SAVE:"<<result_image_folder+test_image_name+".jpg"<<endl;
            imwrite(result_image_folder+test_image_name+".jpg", pyramid.getImageWithObjects());
        }
    }
    test_file.close();
    output_file.close();

    return 0;
}
