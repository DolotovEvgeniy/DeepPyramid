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

Rect DeepPyramid::originalRect2Norm5(const Rect& originalRect, int level, const Size& imgSize)
{
    double longSide=std::max(imgSize.height, imgSize.width);
    Size networkOutputSize=net->outputLayerSize();
    double scale=networkOutputSize.width/(pow(2, (config.numLevels-1-level)/2.0)*longSide);
    return scaleRect(originalRect, scale);
}

Rect DeepPyramid::norm5Rect2Original(const Rect& norm5Rect, int level, const Size& imgSize)
{
    double longSide=std::max(imgSize.height, imgSize.width);
    Size networkOutputSize=net->outputLayerSize();
    double scale=(pow(2, (config.numLevels-1-level)/2.0)*longSide)/networkOutputSize.width;
    return scaleRect(norm5Rect, scale);
}

//Image Pyramid
//
Size DeepPyramid::embeddedImageSize(const Size& imgSize, const int& i)
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

void DeepPyramid::constructImagePyramid(const Mat& img, vector<Mat>& imgPyramid)
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

void DeepPyramid::constructFeatureMapPyramid(const Mat& img, vector<FeatureMap>& maps)
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

//Norm5
//

//
////

//Root-Filter sliding window
//
/*
void DeepPyramid::getNegFeatureVector(int levelIndx, const Rect& rect, Mat& feature)
{
    for(unsigned int k=0;k<maps[levelIndx].size();k++)
    {
        Mat m;
        maps[levelIndx][k](rect).copyTo(m);
        for(int h=0;h<rect.height;h++)
        {
            for(int w=0;w<rect.width;w++)
            {
                feature.at<float>(0,w+h*rect.width+k*rect.area())=m.at<float>(h, w);
            }
        }
    }
}

void DeepPyramid::getPosFeatureVector(const Rect& rect, const Size& size, Mat& feature, const Size& imgSize)
{
    cout<<"rect"<<rect<<endl;
    int bestLevel=chooseLevel(size, rect, imgSize);
    cout<<"level:"<<bestLevel<<endl;
    Rect objectRect=originalRect2Norm5(rect, bestLevel, imgSize);
    cout<<"objectRect:"<<objectRect<<endl;
    if(objectRect.area()==0)
    {
        feature.data=0;
    }
    else
    {
        vector<Mat> norm5Resized;
        for(int j=0;j<256;j++)
        {
            Mat m;
            Mat objectMat=maps[bestLevel][j];
            resize(objectMat(objectRect), m, size);
            norm5Resized.push_back(m);
        }

        for(int w=0;w<size.width;w++)
        {
            for(int h=0;h<size.height;h++)
            {
                for(unsigned int k=0;k<maps.size();k++)
                {
                    int featureIndex=w+h*size.width+k*size.height*size.width;
                    feature.at<float>(0,featureIndex)=norm5Resized[k].at<float>(h,w);
                }
            }
        }
    }
}

*/
void DeepPyramid::detect(const vector<FeatureMap>& maps, vector<BoundingBox>& detectedObjects)
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

void DeepPyramid::calculateOriginalRectangle(vector<BoundingBox>& detectedObjects, const Size& imgSize)
{
    for(unsigned int i=0;i<detectedObjects.size();i++)
    {
        Rect originalRect=norm5Rect2Original(detectedObjects[i].norm5Box, detectedObjects[i].level, imgSize);
        detectedObjects[i].originalImageBox=originalRect;
    }
}

void DeepPyramid::groupOriginalRectangle(vector<BoundingBox>& detectedObjects)
{
    NMSavg nms;
    nms.processBondingBox(detectedObjects,0.2,0.7);
}

void DeepPyramid::detect(const Mat& img, vector<BoundingBox>& objects)
{
    assert(img.channels()==3);
    vector<FeatureMap> maps;
    constructFeatureMapPyramid(img, maps);
    cout<<"filter"<<endl;
    vector<BoundingBox> detectedObjects;
    detect(maps ,detectedObjects);
    cout<<"group rectangle"<<endl;
    calculateOriginalRectangle(detectedObjects, Size(img.cols, img.rows));
    groupOriginalRectangle(detectedObjects);
    objects=detectedObjects;
    cout<<"boundbox regressor: TODO"<<endl;
    cout<<"Object count:"<<detectedObjects.size()<<endl;
}

int DeepPyramid::chooseLevel(const Size& filterSize,const Rect& boundBox, const Size& imgSize)
{
    vector<double> f;
    for(int i=0;i<config.numLevels;i++)
    {
        Rect r=originalRect2Norm5(boundBox, i, imgSize);

        f.push_back(abs(filterSize.width-r.width)+abs(r.height-filterSize.height));
    }
    int bestLevel=distance(f.begin(), min_element(f.begin(), f.end()));

    return bestLevel;
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

/*
void DeepPyramid::extractFeatureVectors(const Mat& img, const Size& filterSize, const vector<Rect>& objectsRect, Mat& features, Mat& labels)
{
    constructFeatureMapPyramid(img, maps);
    cout<<"Cut negatives"<<endl;

    for(int level=0; level<config.numLevels; level++)
    {
        vector<Rect> norm5Objects;
        for(unsigned int j=0; j<norm5Objects.size(); j++)
        {
            Rect objectRect=originalRect2Norm5(objectsRect[j], level, Size(img.cols, img.rows));
            norm5Objects.push_back(objectRect);
        }

        int stepWidth, stepHeight, stride=config.stride;

        stepWidth=((maps[0][0].cols/pow(2,(config.numLevels-1-level)/2.0)-filterSize.width)/stride)+1;
        stepHeight=((maps[0][0].rows/pow(2,(config.numLevels-1-level)/2.0)-filterSize.height)/stride)+1;

        for(int w=0;w<stepWidth;w+=stride)
            for(int h=0;h<stepHeight;h+=stride)
            {
                Point p(stride*w, stride*h);
                Rect sample(p, filterSize);

                bool addNeg=true;
                for(unsigned int objectNum=0; objectNum<norm5Objects.size() ;objectNum++)
                {
                    if(IOU(norm5Objects[objectNum],sample)>0.3)
                    {
                        addNeg=false;
                        break;
                    }
                }

                if(addNeg)
                {
                    Mat feature(1,filterSize.area()*maps[0].size(),CV_32FC1);
                    getNegFeatureVector(level, sample, feature);
                    features.push_back(feature);
                    labels.push_back(NOT_OBJECT);
                }

            }
    }
    cout<<"Cut positive"<<endl;
    for(unsigned int i=0;i<objectsRect.size();i++)
    {
        if(isContain(img, objectsRect[i]))
        {
            Mat feature(1,filterSize.area()*maps[0].size(),CV_32FC1);
            getPosFeatureVector(objectsRect[i], filterSize, feature, Size(img.cols, img.rows));
            if(feature.data)
            {
                features.push_back(feature);
                labels.push_back(OBJECT);
            }
        }
    }

}
*/
DeepPyramid::~DeepPyramid()
{
}
