#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>

#include <deep_pyramid.h>

#include <string>

using namespace cv;
using namespace std;
using namespace caffe;

static const char argsDefs[] =
        "{ | config           |     | Path to configuration file }";

void printHelp(std::ostream& os)
{
    os << "\tUsage: --config=path/to/config.xm" << std::endl;
}

namespace ReturnCode
{
enum
{
    Success = 0,
    ConfigFileNotSpecified = 1,
    ConfigFileNotFound = 2,
    ImageFileNotFound = 3,
    TrainFileNotFound = 4
};
};


class TrainConfiguration
{
public:
    string file_with_train_image;
    string train_image_folder;
    Size filterSize;
    int bootStrapTrainIter;
    int sampleMaxCount;
    string SVMSaveName;

    TrainConfiguration(FileStorage& config)
    {
        config["FileWithTrainImage"]>>file_with_train_image;
        config["TrainImageFolder"]>>train_image_folder;
        config["TrainFilter-size"]>>filterSize;
        config["BootStrapTrainIter"]>>bootStrapTrainIter;
        config["SampleMaxCount"]>>sampleMaxCount;
        config["SVMSaveName"]>>SVMSaveName;
    }

};

Rect ellipseToRect(int major_axis_radius,int minor_axis_radius,double angle,int center_x,int center_y)
{
    double alpha,betta;
    alpha=atan(-major_axis_radius*tan(angle)/minor_axis_radius);
    betta=atan(major_axis_radius/(tan(angle)*minor_axis_radius));
    double xMax=center_x+minor_axis_radius*cos(alpha)*cos(angle);
    xMax-=major_axis_radius*sin(alpha)*sin(angle);
    double xMin=center_x-minor_axis_radius*cos(alpha)*cos(angle);
    xMin+=major_axis_radius*sin(alpha)*sin(angle);
    double yMax=center_y+major_axis_radius*sin(betta)*cos(angle);
    yMax+=minor_axis_radius*cos(betta)*sin(angle);
    double yMin=center_y-major_axis_radius*sin(betta)*cos(angle);
    yMin-=minor_axis_radius*cos(betta)*sin(angle);
    int xSide=fabs(xMax-xMin);
    int ySide=fabs(yMin-yMax);

    return Rect(Point(center_x-ySide/2.0,center_y-xSide/2.0),Size(ySide,xSide));
}

Rect readEllipseToRect(istream& file)
{
    double a, b, h, k, phi;
    file>>a;
    file>>b;
    file>>phi;
    file>>h;
    file>>k;
    int type;
    file>>type;

    return ellipseToRect(a,b,phi,h,k);
}
#define MARGIN_THRESHOLD 1

int main(int argc, char *argv[])
{
    CommandLineParser parser(argc, argv, argsDefs);
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

    DeepPyramid pyramid(config);
    TrainConfiguration trainConfig(config);

    ifstream train_file(trainConfig.file_with_train_image);

    if(train_file.is_open()==false)
    {
        std::cerr << "Test file '" << trainConfig.file_with_train_image
                  << "' not found. Exiting" << std::endl;
        return ReturnCode::TrainFileNotFound;
    }

    Mat features;
    Mat labels;

    string img_path;
    for(int i=0;i<30;i++)
    {
        train_file>>img_path;

        Mat image;
        image=imread(trainConfig.train_image_folder+img_path+".jpg");

        if(!image.data)
        {
            std::cerr << "File '" << trainConfig.train_image_folder+img_path+".jpg"
                      << "' not found. Exiting." << std::endl;
            return ReturnCode::ImageFileNotFound;
        }
        cout<<trainConfig.train_image_folder+img_path+".jpg"<<endl;
        int n;
        train_file>>n;

        vector<Rect> objects;
        for(int i=0;i<n;i++)
        {
            Rect  rect=readEllipseToRect(train_file);
            objects.push_back(rect);
        }

        pyramid.extractFeatureVectors(image, trainConfig.filterSize, objects, features, labels);

        cout<<"Count of features:"<<features.rows<<endl;
    }
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);
    CvSVM* svm=new CvSVM();
    svm->train_auto(features,labels, Mat(),Mat(), params);

    for(int i=0;i<trainConfig.bootStrapTrainIter;i++)
    {
        train_file.close();
        train_file.open(trainConfig.file_with_train_image);
        while(train_file>>img_path)
        {
            Mat image;
            image=imread(trainConfig.train_image_folder+img_path+".jpg");

            if(!image.data)
            {
                std::cerr << "File '" << trainConfig.train_image_folder+img_path+".jpg"
                          << "' not found. Exiting." << std::endl;
                return ReturnCode::ImageFileNotFound;
            }
            cout<<trainConfig.train_image_folder+img_path+".jpg"<<endl;
            int n;
            train_file>>n;

            vector<Rect> objects;
            for(int i=0;i<n;i++)
            {
                Rect  rect=readEllipseToRect(train_file);
                objects.push_back(rect);
            }

            Mat featuresWithOutEasy;
            Mat labelsWithOutEasy;
            for(int i=0;i<features.rows;i++)
            {
                if(labels.at<int>(i,0)==OBJECT || svm->predict(features.row(i))!=labels.at<int>(i,0))
                {
                    featuresWithOutEasy.push_back(features.row(i));
                    labelsWithOutEasy.push_back(labels.at<int>(i,0));
                }
                else
                {
                    if(fabs(svm->predict(features.row(i), true))<MARGIN_THRESHOLD)
                    {
                        featuresWithOutEasy.push_back(features.row(i));
                        labelsWithOutEasy.push_back(labels.at<int>(i,0));
                    }
                }
            }
            featuresWithOutEasy.copyTo(features);
            labelsWithOutEasy.copyTo(labels);

            featuresWithOutEasy.release();
            labelsWithOutEasy.release();


            Mat newFeatures;
            Mat newLabels;
            pyramid.extractFeatureVectors(image, trainConfig.filterSize, objects, newFeatures, newLabels);

            for(int i=0;i<newFeatures.rows;i++)
            {
                if(newLabels.at<int>(i,0)==OBJECT || svm->predict(newFeatures.row(i))!=newLabels.at<int>(i,0))
                {
                    features.push_back(newFeatures.row(i));
                    labels.push_back(newLabels.at<int>(i,0));
                }
                else
                {
                    if(fabs(svm->predict(newFeatures.row(i), true))<MARGIN_THRESHOLD)
                    {
                        features.push_back(newFeatures.row(i));
                        labels.push_back(newLabels.at<int>(i,0));
                    }
                }
            }

            cout<<"Count of features:"<<features.rows<<endl;
        }
        svm->train_auto(features,labels, Mat(),Mat(), params);
        svm->save(("linear_svm"+std::to_string(static_cast<long long>(i))+".xml").c_str());
    }

    config.release();
    train_file.close();
    delete svm;
    return ReturnCode::Success;
}
