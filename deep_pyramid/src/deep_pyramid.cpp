#include <deep_pyramid.h>
#include <nms.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <algorithm>

#include "rectangle_transform.h"

using namespace cv;
using namespace std;
using namespace caffe;

Rect DeepPyramid::norm5Rect2Original(const Rect& norm5Rect, int level, const Size& imgSize) const
{
    double longSide=std::max(imgSize.height, imgSize.width);
    Size networkOutputSize=net->outputLayerSize();
    double scale=(pow(2, (levelCount-1-level)/2.0)*longSide)/networkOutputSize.width;
    return scaleRect(norm5Rect, scale);
}

//Image Pyramid
//
Size DeepPyramid::embeddedImageSize(const Size& imgSize, const int& i) const
{
    Size networkInputSize=net->inputLayerSize();
    Size newImgSize;
    double scale=1/pow(2,(levelCount-1-i)/2.0);
    double aspectRatio=imgSize.height/(double)imgSize.width;
    if(imgSize.height<=imgSize.width)
    {
        newImgSize.width=networkInputSize.width*scale;
        newImgSize.height=newImgSize.width*aspectRatio;
    }
    else
    {
        newImgSize.height=networkInputSize.height*scale;
        newImgSize.width=newImgSize.height/aspectRatio;
    }

    return newImgSize;
}

void DeepPyramid::constructImagePyramid(const Mat& img, vector<Mat>& imgPyramid) const
{
    Size imgSize(img.cols, img.rows);
    for(int level=0; level<levelCount; level++)
    {
        Mat imgAtLevel(net->inputLayerSize(),CV_8UC3, Scalar::all(0));


        Mat resizedImg;
        Size resizedImgSize=embeddedImageSize(imgSize, level);
        resize(img, resizedImg, resizedImgSize);
        resizedImg.copyTo(imgAtLevel(Rect(Point(0,0),resizedImgSize)));
        imgPyramid.push_back(imgAtLevel);
    }
    cout<<"Construct image pyramid..."<<endl;
    cout<<"Status: Success!"<<endl;
}
//
////

void DeepPyramid::constructFeatureMapPyramid(const Mat& img, vector<FeatureMap>& maps) const
{
    vector<Mat> imgPyramid;
    constructImagePyramid(img, imgPyramid);
    for(int i=0; i<levelCount; i++)
    {
        cout<<"Construct feature map ["<<i+1<<"] ..."<<endl;
        FeatureMap map;
        net->processImage(imgPyramid[i], map);
        map.normalize();
        maps.push_back(map);
        cout<<"Status: Success!"<<endl;
    }
}
//
////
void DeepPyramid::detect(const vector<FeatureMap>& maps, vector<BoundingBox>& detectedObjects) const
{
    for(unsigned int i=0;i<rootFilter.size();i++)
        for(unsigned int j=0;j<maps.size();j++)
        {
            vector<BoundingBox> detectedObjectsAtLevel;

            rootFilter[i].processFeatureMap(maps[j], detectedObjectsAtLevel, stride);
            for(unsigned int k=0;k<detectedObjectsAtLevel.size();k++)
            {
                detectedObjectsAtLevel[k].level=j;
                detectedObjects.push_back(detectedObjectsAtLevel[k]);
            }
        }
}
//
////

//Rectangle
//

void DeepPyramid::calculateOriginalRectangle(vector<BoundingBox>& detectedObjects, const Size& imgSize) const
{
    for(unsigned int i=0;i<detectedObjects.size();i++)
    {
        Rect originalRect=norm5Rect2Original(detectedObjects[i].norm5Box, detectedObjects[i].level, imgSize);
        detectedObjects[i].originalImageBox=originalRect;
    }
}

void DeepPyramid::groupRectangle(vector<BoundingBox>& detectedObjects) const
{
    NMSavg nms;
    nms.processBondingBox(detectedObjects,0.2,0.7);
}

DeepPyramid::DeepPyramid(string config)
{
    load(config);
}
void DeepPyramid::load(string config_file)
{
    FileStorage config;

    config.open(config_file, FileStorage::READ);

    if(config.isOpened()==false)
    {
        std::cerr << "File '" << config_file
                  << "' not found. Exiting." << std::endl;
        return;
    }

    string model_file, trained_net_file;

    config["NeuralNetwork-configuration"]>>model_file;
    config["NeuralNetwork-trained-model"]>>trained_net_file;

    net=new NeuralNetwork(model_file, trained_net_file);

    config["NumberOfLevel"]>>levelCount;

    config["Stride"]>>stride;

    string svm_trained_file;
    config["SVM"]>>svm_trained_file;

    Size filterSize;
    config["Filter-size"]>>filterSize;

    CvSVM* classifier=new CvSVM();
    classifier->load(svm_trained_file.c_str());
    rootFilter.push_back(RootFilter(filterSize, classifier));

    //config["BoundingBox-regressor"]>>box_regressor_file;
    //TODO
}

void DeepPyramid::detect(const Mat &img, vector<BoundingBox> &objects, bool isBoundingBoxRegressor) const
{
    CV_Assert(img.channels()==3);
    vector<FeatureMap> maps;
    constructFeatureMapPyramid(img, maps);
    cout<<"filter"<<endl;
    detect(maps, objects);
    cout<<"group rectangle"<<endl;
    calculateOriginalRectangle(objects, Size(img.cols, img.rows));
    groupRectangle(objects);
    if(isBoundingBoxRegressor)
    {
        cout<<"boundbox regressor: TODO"<<endl;
    }
    else
    {
        cout<<"bounding box regressor switch off"<<endl;
    }
    cout<<"Object count:"<<objects.size()<<endl;
}

DeepPyramid::~DeepPyramid()
{
}
