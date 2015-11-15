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

int main(int argc, char *argv[])
{
    Caffe::set_mode(Caffe::CPU);

    string alexnet_model_file=argv[1];
    string alexnet_trained_file=argv[2];

    DeepPyramid pyramid(7,alexnet_model_file, alexnet_trained_file);

    Size filterSize(atoi(argv[3]),atoi(argv[4]));

    ifstream file(argv[5]);
    string path;
    cout<<"train first SVM"<<endl;
    Mat features;
    Mat label;
    int iter=0;
    while(file>>path&&iter<30)
    {
        iter++;
        cout<<path<<endl;
        Mat image;
        image=imread(path+".jpg");
        int n;
        file>>n;
        vector<Rect> objects;
        for(int i=0;i<n;i++)
        {

            double a, b, h, k, phi;
            file>>a;
            file>>b;
            file>>phi;
            file>>h;
            file>>k;
            int type;
            file>>type;

            Rect  rect=ellipseToRect(a,b,phi,h,k);
            objects.push_back(rect);
        }

        pyramid.calculateToNorm5(image);

        int num_level=pyramid.getNumLevel();

        for(int level=0; level<num_level; level++)
        {
            vector<Rect> norm5Object;
            for(int j=0;j<n;j++)
            {
                Rect imagePyramidBounigBox=pyramid.boundingBoxAtLevel(level,objects[j]);
                Rect objectRect=pyramid.getNorm5RectByOriginal(imagePyramidBounigBox);
                norm5Object.push_back(objectRect);
            }

            int stepWidth, stepHeight, stride=2;

            int norm5Side=pyramid.norm5SideLength();
            stepWidth=((norm5Side/pow(2,(num_level-1-level)/2.0)-filterSize.width)/stride)+1;
            stepHeight=((norm5Side/pow(2,(num_level-1-level)/2.0)-filterSize.height)/stride)+1;

            for(int w=0;w<stepWidth;w+=stride)
                for(int h=0;h<stepHeight;h+=stride)
                {
                    Point p(stride*w,stride*h);
                    Rect sample(p,Size(7,11));

                    bool addNeg=true;
                    for(int objectNum=0;objectNum<n;objectNum++)
                    {
                        if(IOU(norm5Object[objectNum],sample)>0.3)
                        {
                            addNeg=false;
                            break;
                        }
                    }
                    if(addNeg)
                    {
                        Mat feature=pyramid.getFeatureVector(level,p,Size(7,11));
                        features.push_back(feature);
                        label.push_back(-1);
                    }

                }
        }

        for(int i=0;i<n;i++)
        {
            int bestLevel=pyramid.chooseLevel(filterSize,objects[i]);

            Rect imagePyramidBounigBox=pyramid.boundingBoxAtLevel(bestLevel,objects[i]);
            Rect objectRect=pyramid.getNorm5RectByOriginal(imagePyramidBounigBox);
            vector<Mat> norm5;
            for(int j=0;j<256;j++)
            {
                Mat m;
                Mat objectMat=pyramid.getNorm5(bestLevel,j);
                resize(objectMat(objectRect),m,filterSize);
                norm5.push_back(m);
            }

            cv::Mat feature(1,filterSize.height*filterSize.width*norm5.size(),CV_32FC1);
            for(int w=0;w<filterSize.width;w++)
            {
                for(int h=0;h<filterSize.height;h++)
                {
                    for(unsigned int k=0;k<norm5.size();k++)
                    {
                        int featureIndex=w+h*filterSize.width+k*filterSize.height*filterSize.width;
                        feature.at<float>(0,featureIndex)=norm5[k].at<float>(h,w);
                    }
                }
            }
            features.push_back(feature);
            label.push_back(1);
        }
    }
    file.close();
    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::POLY;
    params.degree=3;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 500, 1e-6);
    CvSVM svm;
    svm.train(features,label, Mat(),Mat(),params);
    pyramid.addRootFilter(filterSize, &svm);
    int retrainCount=atoi(argv[6]);
    for(int retrainIter=0; retrainIter<retrainCount; retrainIter++)
    {
        file.open(argv[5]);
        while(file>>path)
        {
            cout<<path<<endl;
            Mat image;
            image=imread(path+".jpg");
            int n;
            file>>n;
            vector<Rect> objects;
            for(int i=0;i<n;i++)
            {

                double a, b, h, k, phi;
                file>>a;
                file>>b;
                file>>phi;
                file>>h;
                file>>k;
                int type;
                file>>type;

                Rect  rect=ellipseToRect(a,b,phi,h,k);
                objects.push_back(rect);
            }

            vector<ObjectBox> detectedObject;
            detectedObject=pyramid.detect(image);

            for(unsigned int i=0;i<detectedObject.size();i++)
            {
                bool addFalsePositive=true;
                for(int objectNum=0;objectNum<n;objectNum++)
                {
                    if(IOU(objects[objectNum],detectedObject[i].originalImageBox)>0.4)
                    {
                        addFalsePositive=false;
                        break;
                    }
                }
                if(addFalsePositive)
                {
                    int bestLevel=pyramid.chooseLevel(filterSize,detectedObject[i].originalImageBox);

                    Rect imagePyramidBounigBox=pyramid.boundingBoxAtLevel(bestLevel,detectedObject[i].originalImageBox);
                    Rect objectRect=pyramid.getNorm5RectByOriginal(imagePyramidBounigBox);
                    vector<Mat> norm5;
                    for(int j=0;j<256;j++)
                    {
                        Mat m;
                        Mat objectMat=pyramid.getNorm5(bestLevel,j);
                        resize(objectMat(objectRect),m,filterSize);
                        norm5.push_back(m);
                    }

                    cv::Mat feature(1,filterSize.height*filterSize.width*norm5.size(),CV_32FC1);
                    for(int w=0;w<filterSize.width;w++)
                    {
                        for(int h=0;h<filterSize.height;h++)
                        {
                            for(unsigned int k=0;k<norm5.size();k++)
                            {
                                int featureIndex=w+h*filterSize.width+k*filterSize.height*filterSize.width;
                                feature.at<float>(0,featureIndex)=norm5[k].at<float>(h,w);
                            }
                        }
                    }
                    features.push_back(feature);
                    label.push_back(-1);
                }
            }
        }
        file.close();
        svm.train(features,label, Mat(),Mat(),params);
        pyramid.clearFilter();
        pyramid.addRootFilter(filterSize,&svm);
    }




    return 0;
}
