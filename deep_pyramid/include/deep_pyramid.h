#ifndef DEEP_PYRAMID_H
#define DEEP_PYRAMID_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>
#include <utility>
#include <stdio.h>

#include <bounding_box.h>
#include "neural_network.h"
#include "root_filter.h"
#include <bounding_box_regressor.h>

#define TIMER_START(name) int64 t_##name = getTickCount()
#define TIMER_END(name) printf("TIMER_" #name ":\t%6.2fms\n", \
    1000.f * ((getTickCount() - t_##name) / getTickFrequency()))

#define OBJECT 1
#define NOT_OBJECT -1

class DeepPyramidConfiguration
{
private:
    std::string model_file;
    std::string trained_net_file;

    int numLevels;

    cv::Scalar objectRectangleColor;

    std::string svm_trained_file;
    cv::Size filterSize;

    int stride;

    DeepPyramidConfiguration(cv::FileStorage& configFile);

    friend class DeepPyramid;
};
class DeepPyramid
{
public:
    DeepPyramid(cv::FileStorage& configFile);

    ~DeepPyramid();

    void detect(const cv::Mat& img, std::vector<BoundingBox>& objects) const;

    void constructFeatureMapPyramid(const cv::Mat& img, std::vector<FeatureMap>& maps) const;
    double levelScale;
    int  levelCount;
private:


    DeepPyramidConfiguration config;

    std::vector<RootFilter> rootFilter;

    NeuralNetwork* net;
    BoundingBoxRegressor regressor;

    //Image Pyramid
    cv::Size embeddedImageSize(const cv::Size& img, const int& level) const;
    void constructImagePyramid(const cv::Mat& img, std::vector<cv::Mat>& imgPyramid) const;

    void detect(const std::vector<FeatureMap>& maps,std::vector<BoundingBox>& detectedObjects) const;

    //rename
    cv::Rect norm5Rect2Original(const cv::Rect& norm5Rect, int level, const cv::Size& imgSize) const;
    void calculateOriginalRectangle(std::vector<BoundingBox>& detectedObjects, const cv::Size& imgSize) const;
    void groupRectangle(std::vector<BoundingBox>& detectedObjects) const;
};
#endif
