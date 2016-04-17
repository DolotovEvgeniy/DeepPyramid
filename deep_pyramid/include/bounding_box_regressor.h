// Copyright 2016 Dolotov Evgeniy

#ifndef DEEP_PYRAMID_INCLUDE_BOUNDING_BOX_REGRESSOR_H_
#define DEEP_PYRAMID_INCLUDE_BOUNDING_BOX_REGRESSOR_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <string>
#include "../include/bounding_box.h"

class BoundingBoxRegressor {
 public:
    void save(const std::string filename);
    void load(const std::string filename);
    void processBoundingBoxes(std::vector<BoundingBox>& objects);
 private:
    void regressBox(BoundingBox& object);
    cv::Mat xWeights;
    cv::Mat yWeights;
    cv::Mat widthWeights;
    cv::Mat heightWeights;
};

#endif  // DEEP_PYRAMID_INCLUDE_BOUNDING_BOX_REGRESSOR_H_
