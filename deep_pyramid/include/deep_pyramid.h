// Copyright 2016 Dolotov Evgeniy

#ifndef DEEP_PYRAMID_INCLUDE_DEEP_PYRAMID_H_
#define DEEP_PYRAMID_INCLUDE_DEEP_PYRAMID_H_

#include <vector>
#include <utility>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "bounding_box.h"
#include "neural_network.h"
#include "bounding_box_regressor.h"
#include "feature_map_svm.h"

const double NEGATIVE_THRESHOLD = 0.3;
const double POSITIVE_THRESHOLD = 0.7;

const double BOX_THRESHOLD = 0.2;
const double CONFIDENCE_THRESHOLD = 0.7;

class DeepPyramid {  
private:
    double levelScale;
    size_t  levelCount;
    int stride;
    std::vector<FeatureMapSVM> rootFilter;
    NeuralNetwork* net;
    BoundingBoxRegressor regressor;

private:
    cv::Size embeddedImageSize(const cv::Size& img, const int& level) const;
    void constructImagePyramid(const cv::Mat& img,
                               std::vector<cv::Mat>& imgPyramid) const;

    void constructFeatureMapPyramid(const std::vector<cv::Mat>& imgPyramid,
                                    std::vector<FeatureMap>& maps) const;
    void processFeatureMap(const int& filterIndx, const FeatureMap& map,
                           std::vector<BoundingBox>& detectedObjects) const;

    void detect(const std::vector<FeatureMap>& maps,
                std::vector<BoundingBox>& detectedObjects) const;

    double imageScale(const int& level) const;
    cv::Rect norm5Rect2Original(const cv::Rect& norm5Rect, const int& level,
                                const cv::Size& imgSize) const;
    cv::Rect originalRect2Norm5(const cv::Rect& originalRect, const int& level,
                                const cv::Size& imgSize) const;
    int chooseLevel(const cv::Size& filterSize, const cv::Rect& boundBox,
                    const cv::Size& imgSize) const;
    void calculateOriginalRectangle(std::vector<BoundingBox>& detectedObjects,
                                    const cv::Size& imgSize) const;

    void groupRectangle(std::vector<BoundingBox>& detectedObjects) const;

public:

    DeepPyramid(const cv::FileStorage& config);

    void detect(const cv::Mat& img, std::vector<cv::Rect>& objects,
                std::vector<float>& confidence,
                bool isBoundingBoxRegressor = true) const;
    void detect(const cv::Mat& img, std::vector<BoundingBox>& objects,
                bool isBoundingBoxRegressor = true) const;
    void extractFeatureMap(const cv::Mat& img, std::vector<cv::Rect>& objects,
                           const cv::Size& size, std::vector<FeatureMap>& omaps,
                           std::vector<FeatureMap>& nmaps);
};
#endif  // DEEP_PYRAMID_INCLUDE_DEEP_PYRAMID_H_
