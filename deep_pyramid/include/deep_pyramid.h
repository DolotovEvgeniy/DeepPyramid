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

#define OBJECT 1
#define NOT_OBJECT -1

class DeepPyramid
{
public:
    void changeRootFilter(FeatureMapSVM svm, cv::Size filterSize);
    DeepPyramid(std::string model_file, std::string trained_net_file,
                std::vector<std::string> svm_file, std::vector<cv::Size> svmSize,
                int levelCount=7, int stride=1);
    DeepPyramid(cv::FileStorage config);
    ~DeepPyramid();

    void detect(const cv::Mat& img, std::vector<cv::Rect>& objects, std::vector<float>& confidence, bool isBoundingBoxRegressor=true) const;
    void detect(const cv::Mat& img, std::vector<BoundingBox>& objects, bool isBoundingBoxRegressor=true) const;
    void extractObjectsFeatureMap(const cv::Mat& img, std::vector<cv::Rect>& objects, std::vector<FeatureMap>& maps);
    void extractNotObjectsFeatureMap(const cv::Mat& img, std::vector<cv::Rect>& objects, cv::Size size, std::vector<FeatureMap>& maps);

private:
    double levelScale;
    int  levelCount;
    int stride;
    void constructFeatureMapPyramid(const cv::Mat& img, std::vector<FeatureMap>& maps) const;
    std::vector<RootFilter> rootFilter;

    NeuralNetwork* net;
    BoundingBoxRegressor regressor;

    cv::Size embeddedImageSize(const cv::Size& img, const int& level) const;

    void constructImagePyramid(const cv::Mat& img, std::vector<cv::Mat>& imgPyramid) const;

    void detect(const std::vector<FeatureMap>& maps,std::vector<BoundingBox>& detectedObjects) const;

    cv::Rect norm5Rect2Original(const cv::Rect& norm5Rect, int level, const cv::Size& imgSize) const;
    cv::Rect originalRect2Norm5(const cv::Rect& originalRect, int level, const cv::Size& imgSize) const;
    int chooseLevel(const cv::Size& filterSize, const cv::Rect& boundBox, const cv::Size& imgSize) const;
    void calculateOriginalRectangle(std::vector<BoundingBox>& detectedObjects, const cv::Size& imgSize) const;
    void groupRectangle(std::vector<BoundingBox>& detectedObjects) const;
};
#endif
