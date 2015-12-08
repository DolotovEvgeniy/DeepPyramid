#ifndef DEEP_PYRAMID_H
#define DEEP_PYRAMID_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <vector>
#include <utility>


#include <stdio.h>
#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
    1000.f * ((getTickCount() - t_##name) / getTickFrequency()))

#define OBJECT 1
#define NOT_OBJECT -1

double IOU(const cv::Rect& r1, const cv::Rect& r2);

class ObjectBox
{
public:
    double confidence;
    int level;
    cv::Rect norm5Box;
    cv::Rect originalImageBox;
    bool operator< (ObjectBox object)
    {
        return confidence<object.confidence;
    }
};

//friend class DeepPyramid
class DeepPyramidConfiguration
{
public:
    std::string model_file;
    std::string trained_net_file;

    unsigned int numLevels;

    cv::Scalar objectRectangleColor;

    std::string svm_trained_file;
    cv::Size filterSize;

    unsigned int stride;

    DeepPyramidConfiguration(std::string deep_pyramid_config);
};

class DeepPyramid
{
public:
    DeepPyramid(std::string deep_pyramid_config);

    void extractFeatureVectors(const cv::Mat& img, const std::vector<cv::Rect>& objectsRect,cv::Mat& features, cv::Mat& labels);

    DeepPyramidConfiguration config;

    void detect(const cv::Mat& img, std::vector<ObjectBox>& objects);

    void getNegFeatureVector(int levelIndx, const cv::Rect& rect, cv::Mat& feature);

    int chooseLevel(const cv::Size& filterSize, const cv::Rect& boundBox);

    //результат через параметры
    void getPosFeatureVector(const cv::Rect& rect, const cv::Size& size, cv::Mat& feature);
    //cv::Mat originalImg;
    //в конфиге есть
    //unsigned int num_levels;
    std::vector<cv::Mat> imagePyramid;
    std::vector< std::vector<cv::Mat> > max5;
    std::vector< std::vector<cv::Mat> > norm5;
    std::vector<std::pair<cv::Size, CvSVM*> > rootFilter;

    //std::vector<ObjectBox> detectedObjects;

    caffe::shared_ptr<caffe::Net<float> > net;

    //Image Pyramid

    //rename
    cv::Size calculateLevelPyramidImageSize(const cv::Mat& img, int level);
    //результат через параметры, переименовать  createImageAtPyrammidLevel?
    cv::Mat createLevelPyramidImage(const cv::Mat& img, int level);
    void createImagePyramid(const cv::Mat& img);

    //NeuralNet
    //delete
    cv::Mat convertToFloat(const cv::Mat& img);
    void fillNeuralNetInput(int level);

    //rename, результат через параметры
    std::vector<cv::Mat> wrapNetOutputLayer();
    void calculateNet();
    //calculateImageRepresentation()
    void calculateNetAtLevel(int level);

    //Max5
    //результат через параметры
    std::vector<cv::Mat> createLevelPyramidMax5(int level);
    void createMax5Pyramid();

    //Norm5
    std::vector<cv::Mat> createLevelPyramidNorm5(int level);
    void createNorm5Pyramid();
    void calculateToNorm5(const cv::Mat& img);

    //Root-Filter sliding window
    void rootFilterAtLevel(int rootFilterIndx, int levelIdx);//, int stride);
    //rename private: detect()
    void rootFilterConvolution();

    //Rectangle transform
    cv::Rect originalRect2Norm5(const cv::Rect& originalRect, int level, const cv::Size& imgSize);
    //rename
    cv::Rect norm5Rect2Original(const cv::Rect& norm5Rect, int level, const cv::Size& imgSize);
    void calculateOriginalRectangle();
    void groupOriginalRectangle();

    //Rectangle transform ARTICLE
    cv::Rect getRectByNorm5Pixel_ARTICLE(cv::Point point);
    cv::Rect getRectByNorm5Rect_ARTICLE(cv::Rect rect);
    cv::Rect getNorm5RectByOriginal_ARTICLE(cv::Rect originalRect);
    int centerConformity;
    int boxSideConformity;
    void clear();


};
#endif // DEEP_PYRAMID_H
