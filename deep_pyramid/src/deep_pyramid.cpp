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

#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
    1000.f * ((getTickCount() - t_##name) / getTickFrequency()))

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
    cout<<"Create image pyramid..."<<endl;
    for(int level=0; level<levelCount; level++)
    {
        Mat imgAtLevel(net->inputLayerSize(),CV_8UC3, Scalar::all(0));


        Mat resizedImg;
        Size resizedImgSize=embeddedImageSize(imgSize, level);
        resize(img, resizedImg, resizedImgSize);
        resizedImg.copyTo(imgAtLevel(Rect(Point(0,0),resizedImgSize)));
        imgPyramid.push_back(imgAtLevel);
    }
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

void DeepPyramid::detect(const Mat& img, vector<Rect>& detectedObjects, vector<float>& confidence, bool isBoundingBoxRegressor) const
{
    CV_Assert(img.channels()==3);
    vector<FeatureMap> maps;
    constructFeatureMapPyramid(img, maps);
    cout<<img.cols<<","<<img.rows<<","<<rootFilter.size()<<endl;
    vector<BoundingBox> objects;
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
    for(unsigned int i=0;i<objects.size();i++)
    {
        detectedObjects.push_back(objects[i].originalImageBox);
        confidence.push_back(objects[i].confidence);
    }
}

DeepPyramid::DeepPyramid(string model_file, string trained_net_file,
                         vector<string> svm_file, vector<Size> svmSize,
                         int _levelCount, int _stride)
{
    net=new NeuralNetwork(model_file, trained_net_file);
    levelCount=_levelCount;
    stride=_stride;
    for(unsigned int i=0;i<svm_file.size();i++)
    {
        rootFilter.push_back(RootFilter(svmSize[i], svm_file[i]));
    }
}
DeepPyramid::DeepPyramid(FileStorage config)
{
    string model_file;
    string trained_net_file;
    config["net"]>>model_file;
    config["weights"]>>trained_net_file;
    net=new NeuralNetwork(model_file, trained_net_file);

    config["number_of_levels"]>>levelCount;

    string svm_trained_file;
    config["svm"]>>svm_trained_file;
    Size filterSize;
    config["filter_size"]>>filterSize;
    RootFilter filter(filterSize, svm_trained_file);
    rootFilter.push_back(filter);
    cout<<"HERE!!!"<<endl;
    config["stride"]>>stride;
    cout<<"HERE!!!"<<endl;
}

void DeepPyramid::detect(const Mat &img, vector<BoundingBox> &objects, bool isBoundingBoxRegressor) const
{
    CV_Assert(img.channels()==3);
    vector<FeatureMap> maps;
    cout<<"here!"<<endl;
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

Rect DeepPyramid::originalRect2Norm5(const Rect& originalRect, int level, const Size& imgSize) const
{

    double longSide=std::max(imgSize.height, imgSize.width);

    Size networkOutputSize=net->outputLayerSize();
    double scale=networkOutputSize.width/(pow(levelScale, levelCount-level-1)*longSide);
    return scaleRect(originalRect, scale);
}

int DeepPyramid::chooseLevel(const Size& filterSize, const Rect& boundBox, const Size& imgSize) const
{
    vector<double> f;
    for(int i=0;i<levelCount;i++)
    {
        Rect r=originalRect2Norm5(boundBox, i, imgSize);

        f.push_back(abs(filterSize.width-r.width)+abs(r.height-filterSize.height));
    }
    int bestLevel=distance(f.begin(), min_element(f.begin(), f.end()));

    return bestLevel;
}

void DeepPyramid::changeRootFilter(FeatureMapSVM svm, Size filterSize)
{
    rootFilter.clear();
  //  rootFilter.push_back(RootFilter(filterSize, svm));
}

void DeepPyramid::extractNotObjectsFeatureMap(const Mat &img, vector<Rect> &objects, Size size, vector<FeatureMap> &maps)
{
    Size imgSize(img.cols, img.rows);

    vector<FeatureMap> featureMaps;
    constructFeatureMapPyramid(img, featureMaps);
    for(int i=0;i<levelCount;i++)
    {
        vector<Rect> objectsAtLevel;
        for(unsigned int obj=0;obj<objects.size();obj++)
        {
            objectsAtLevel.push_back(originalRect2Norm5(objects[obj], i, imgSize));
        }

        Size mapSize=featureMaps[i].size();
        for(int w=0;w<mapSize.width-size.width;w++)
            for(int h=0;h<mapSize.height-size.height;h++)
            {
                bool isNegative=true;
                for(unsigned int obj=0;obj<objects.size();obj++)
                {
                    if(IOU(Rect(Point(w,h), size), objectsAtLevel[obj])>0.7)
                        isNegative=false;
                }
                FeatureMap map;
                if(isNegative)
                {
                    featureMaps[i].extractFeatureMap(Rect(Point(w,h), size), map);
                    maps.push_back(map);
                }
            }
    }
}


void DeepPyramid::extractObjectsFeatureMap(const Mat& image, vector<Rect> &objects, vector<FeatureMap> &maps)
{
    vector<FeatureMap> featureMaps;
    constructFeatureMapPyramid(image, featureMaps);
    for(unsigned int i=0;i<objects.size();i++)
    {
        Size imgSize(image.cols, image.rows);
        int level=chooseLevel(rootFilter[0].getFilterSize(), objects[i], imgSize);
        Rect norm5Rect=originalRect2Norm5(objects[i], level, imgSize);
        FeatureMap map;
        featureMaps[level].extractFeatureMap(norm5Rect, map);
        maps.push_back(map);
    }
}
