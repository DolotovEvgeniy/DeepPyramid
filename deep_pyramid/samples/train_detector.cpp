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

Mat diagMatrix(int n, float scalar)
{
    Mat I(n, n, CV_32FC1, Scalar::all(0));

    for(int i=0;i<n;i++)
    {
        I.at<float>(i,i)=scalar;
    }
    return I;
}

Point centerOfRect(Rect rect)
{
    return Point(rect.x+rect.width/2.0, rect.y+rect.height/2.0);
}

void readImg(string img_path, istream& file, Mat& img)
{
    cout<<img_path<<endl;
    img=imread(img_path+".jpg");
}

int readObjectCount(istream& file)
{
    int n;
    file>>n;
    return n;
}

void readObjectRect(istream& file, int n, vector<Rect>& objects)
{
    for(int i=0;i<n;i++)
    {
        Rect  rect=readEllipseToRect(file);
        objects.push_back(rect);
    }
}

int main(int argc, char *argv[])
{

    string config_file=argv[1];
    DeepPyramid pyramid(config_file, DeepPyramidMode::TRAIN);

    FileStorage config(config_file, FileStorage::READ);
    Size filterSize;
    config["TrainFilter-size"]>>filterSize;

    string train_file_path;
    config["FileWithTrainImage"]>>train_file_path;

    string train_image_folder;
    config["TrainImageFolder"]>>train_image_folder;

    ifstream train_file(train_file_path);
    string img_path;

    cout<<"train start SVM"<<endl;
    Mat features;
    Mat label;
    int iter=0;
    int maxStartTrainIter;
    config["StartSVMTrainIter"]>>maxStartTrainIter;
    while(train_file>>img_path && iter<maxStartTrainIter)
    {
        iter++;
        Mat image;
        readImg(train_image_folder+img_path, train_file, image);

        int n=readObjectCount(train_file);

        vector<Rect> objects;
        readObjectRect(train_file, n, objects);

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
                    Rect sample(p,filterSize);

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
                        Mat feature(1,filterSize.width*filterSize.height*pyramid.getNorm5ChannelsCount(),CV_32FC1);
                        pyramid.getFeatureVector(level,p, filterSize, feature);
                        features.push_back(feature);
                        label.push_back(NOT_OBJECT);
                    }

                }
        }

        for(int i=0;i<n;i++)
        {
            features.push_back(pyramid.getFeatureVector(objects[i],filterSize));
            label.push_back(OBJECT);
        }
    }

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;

    string SVMType;
    config["SVMKernelType"]>>SVMType;
    if(SVMType=="LINEAR")
    {
        params.kernel_type = CvSVM::LINEAR;
    }
    else
    {
        if(SVMType=="POLY")
        {
            params.kernel_type = CvSVM::POLY;
            config["SVMDegree"]>>params.degree;
        }
    }

    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 500, 1e-6);
    CvSVM svm;

    svm.train(features,label, Mat(),Mat(),params);

    pyramid.addRootFilter(filterSize, &svm);

    int countRetrainIter;
    config["HardNegTrainIter"]>>countRetrainIter;
    int retrainIter=0;
    int maxSampleCount;
    config["SampleMaxCount"]>>maxSampleCount;
    while(retrainIter<countRetrainIter && features.cols<maxSampleCount)
    {
        train_file.clear();
        train_file.seekg(0, ios_base::beg);

        while(train_file>>img_path)
        {
            Mat image;
            readImg(train_image_folder+img_path, train_file, image);

            int n=readObjectCount(train_file);

            vector<Rect> objects;
            readObjectRect(train_file, n, objects);

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
                    features.push_back(pyramid.getFeatureVector(detectedObject[i].originalImageBox,filterSize));
                    label.push_back(NOT_OBJECT);
                }
            }

            for(int i=0;i<n;i++)
            {
                features.push_back(pyramid.getFeatureVector(objects[i],filterSize));
                label.push_back(OBJECT);
            }
        }

        pyramid.clearFilter();

        svm.train(features,label, Mat(),Mat(),params);
        pyramid.addRootFilter(filterSize,&svm);
        retrainIter++;
    }
    string svm_name;
    config["SVMSaveName"]>>svm_name;
    svm.save((svm_name+".xml").c_str());
    features.release();
    label.release();

//    vector< pair<Rect, Rect> > boundingBoxRegressorTrainData;
//    train_file.clear();
//    train_file.seekg(0, ios_base::beg);
//    while(train_file>>img_path && boundingBoxRegressorTrainData.size() < 100)
//    {
//        cout<<img_path<<endl;
//        Mat image;
//        image=imread(train_image_folder+img_path+".jpg");
//        int n;
//        train_file>>n;
//        vector<Rect> objects;
//        for(int i=0;i<n;i++)
//        {
//            Rect  rect=readEllipseToRect(train_file);
//            objects.push_back(rect);
//        }

//        vector<ObjectBox> detectedObject;
//        detectedObject=pyramid.detect(image);

//        for(unsigned int i=0;i<detectedObject.size();i++)
//            for(unsigned int j=0;j<objects.size();j++)
//            {
//                if(IOU(objects[j],detectedObject[i].originalImageBox)>0.6)
//                {
//                    pair< Rect, Rect> group;
//                    group.first=objects[j];
//                    group.second=detectedObject[i].originalImageBox;
//                }
//            }
//    }

//    Mat Tx;
//    Mat Ty;
//    Mat Tw;
//    Mat Th;
//    Mat X;

//    for(unsigned int i=0;i<boundingBoxRegressorTrainData.size();i++)
//    {
//        Rect G=boundingBoxRegressorTrainData[i].first;
//        Rect P=boundingBoxRegressorTrainData[i].second;

//        float tx=(centerOfRect(G).x-centerOfRect(P).x)/(double)P.width;
//        float ty=(centerOfRect(G).y-centerOfRect(P).y)/(double)P.height;
//        float tw=log(G.width/(double)P.width);
//        float th=log(G.height/(double)P.height);
//        Tx.push_back(tx);
//        Ty.push_back(ty);
//        Tw.push_back(tw);
//        Th.push_back(th);
//        Mat boundBoxFeature=pyramid.getFeatureVector(P,filterSize);
//        X.push_back(boundBoxFeature);
//    }

//    Mat XT;

//    transpose(X, XT);
//    Mat XTX;
//    XTX=XT*X;
//    X.release();
//    float A=100;
//    Mat I=diagMatrix(XTX.cols, A);
//    XTX+=I;
//    I.release();
//    Mat revXTX;
//    invert(XTX,revXTX);

//    XTX.release();

//    Mat Wx,Wy,Ww,Wh;

//    Mat XTTx;
//    XTTx=XT*Tx;

//    Wx=revXTX*XTTx;

//    Mat XTTy;

//    XTTy=XT*Ty;

//    Wy=revXTX*XTTy;

//    Mat XTTw;

//    XTTw=XT*Tw;

//    Ww=revXTX*XTTw;

//    Mat XTTh;

//    XTTh=XT*Th;

//    Wh=revXTX*XTTh;
train_file.close();
config.release();
    return 0;
}
