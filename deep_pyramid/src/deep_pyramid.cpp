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
    double scale=(pow(2, (config.numLevels-1-level)/2.0)*longSide)/networkOutputSize.width;
    return scaleRect(norm5Rect, scale);
}

//Image Pyramid
//
Size DeepPyramid::embeddedImageSize(const Size& imgSize, const int& i) const
{
    Size networkInputSize=net->inputLayerSize();
    Size newImgSize;
    double scale=1/pow(2,(config.numLevels-1-i)/2.0);
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
    for(int level=0; level<config.numLevels; level++)
    {
        Mat imgAtLevel(net->inputLayerSize(),CV_8UC3, Scalar::all(0));


        Mat resizedImg;
        Size resizedImgSize=embeddedImageSize(imgSize, level);
        resize(img, resizedImg, resizedImgSize);
        resizedImg.copyTo(imgAtLevel(Rect(Point(0,0),resizedImgSize)));
        imgPyramid.push_back(imgAtLevel);
    }
    cout<<"Create image pyramid..."<<endl;
    cout<<"Status: Success!"<<endl;
}
//
////

void DeepPyramid::constructFeatureMapPyramid(const Mat& img, vector<FeatureMap>& maps) const
{
    vector<Mat> imgPyramid;
    constructImagePyramid(img, imgPyramid);
    for(int i=0; i<config.numLevels; i++)
    {
        FeatureMap map;
        net->processImage(imgPyramid[i], map);
        map.normalize();
        maps.push_back(map);
    }
}
//
////
void DeepPyramid::detect(const vector<FeatureMap>& maps, vector<BoundingBox>& detectedObjects) const
{
    for(unsigned int i=0;i<rootFilter.size();i++)
        for(unsigned int j=0;j<maps.size();j++)
        {
            std::vector<cv::Rect> detectedRect;
            std::vector<double> confidence;
            rootFilter[i].processFeatureMap(maps[j], detectedRect, confidence);
            for(unsigned int k=0;k<detectedRect.size();k++)
            {
                BoundingBox object;
                object.confidence=confidence[k];
                object.level=j;
                object.norm5Box=detectedRect[k];
                detectedObjects.push_back(object);
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

void DeepPyramid::detect(const Mat& img, vector<BoundingBox>& objects) const
{
    CV_Assert(img.channels()==3);
    vector<FeatureMap> maps;
    constructFeatureMapPyramid(img, maps);
    cout<<"filter"<<endl;
    detect(maps, objects);
    cout<<"group rectangle"<<endl;
    calculateOriginalRectangle(objects, Size(img.cols, img.rows));
    groupRectangle(objects);
    cout<<"boundbox regressor: TODO"<<endl;
    cout<<"Object count:"<<objects.size()<<endl;
}

DeepPyramid::DeepPyramid(FileStorage& configFile) : config(configFile)
{
    net=new NeuralNetwork(config.model_file, config.trained_net_file);

    CvSVM* classifier=new CvSVM();
    classifier->load(config.svm_trained_file.c_str());
    rootFilter.push_back(RootFilter(config.filterSize, classifier));
}


DeepPyramidConfiguration::DeepPyramidConfiguration(FileStorage& configFile)
{
    configFile["NeuralNetwork-configuration"]>>model_file;
    configFile["NeuralNetwork-trained-model"]>>trained_net_file;
    configFile["NumberOfLevel"]>>numLevels;

    configFile["Stride"]>>stride;
    configFile["ObjectColor"]>>objectRectangleColor;

    configFile["SVM"]>>svm_trained_file;
    configFile["Filter-size"]>>filterSize;
}

DeepPyramid::~DeepPyramid()
{
}
