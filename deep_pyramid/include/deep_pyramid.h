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

double IOU(cv::Rect r1,cv::Rect r2);

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
enum DeepPyramidMode {DETECT, TRAIN, TEST};

class DeepPyramidConfiguration
{
public:
    DeepPyramidMode mode;

    std::string model_file;
    std::string trained_net_file;

    int numLevels;

    cv::Scalar objectRectangleColor;

    std::string svm_trained_file;
    cv::Size filterSize;

    int stride;
    DeepPyramidConfiguration(){};
    DeepPyramidConfiguration(std::string deep_peramid_config, DeepPyramidMode mode);
};

class DeepPyramid
{
public:
    DeepPyramid(std::string detector_config, DeepPyramidMode mode);
    DeepPyramid(int num_levels, const std::string& model_file,
                const std::string& trained_file);
    int getLevelCount();
    cv::Size norm5Size();
    void cutFeatureFromImage(const cv::Mat& img, const std::vector<cv::Rect>& objectsRect,cv::Mat& features, cv::Mat& labels);
private:
    int getNorm5ChannelsCount();

    DeepPyramidConfiguration config;
    void addRootFilter(cv::Size filterSize, CvSVM* classifier);
    void setImg(const cv::Mat& img);
    std::vector<ObjectBox> detect(cv::Mat img);
    cv::Mat getImageWithObjects();
    void getFeatureVector(int levelIndx, cv::Point position, cv::Size size, cv::Mat& feature);


    int chooseLevel(cv::Size filterSize, cv::Rect boundBox);
    cv::Mat getNorm5(int level, int channel);

    cv::Mat getFeatureVector(cv::Rect rect, cv::Size size);
    cv::Mat originalImg;
    unsigned int num_levels;
    std::vector<cv::Mat> imagePyramid;
    std::vector< std::vector<cv::Mat> > max5;
    std::vector< std::vector<cv::Mat> > norm5;
    std::vector<std::pair<cv::Size, CvSVM*> > rootFilter;

    std::vector<ObjectBox> detectedObjects;

    caffe::shared_ptr<caffe::Net<float> > net_;
    int inputDimension;
    int num_channels;

    //Image Pyramid
    cv::Size calculateLevelPyramidImageSize(int level);
    cv::Mat createLevelPyramidImage(int level);
    void createImagePyramid();

    //NeuralNet
    cv::Mat convertToFloat(const cv::Mat& img);
    void fillNeuralNetInput(int level);
    std::vector<cv::Mat> wrapNetOutputLayer();
    void calculateNet();
    void calculateNetAtLevel(int level);

    //Max5
    std::vector<cv::Mat> createLevelPyramidMax5(int level);
    void createMax5Pyramid();

    //Norm5
    std::vector<cv::Mat> createLevelPyramidNorm5(int level);
    void createNorm5Pyramid();
    void calculateToNorm5(const cv::Mat& img);

    //Root-Filter sliding window
    void rootFilterAtLevel(int rootFilterIndx, int levelIndx, int stride);
    void rootFilterConvolution();

    //Rectangle transform
    cv::Rect getNorm5RectAtLevelByOriginal(cv::Rect originalRect, int level);
    cv::Rect getOriginalRectByNorm5AtLevel(cv::Rect norm5Rect, int level);
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
