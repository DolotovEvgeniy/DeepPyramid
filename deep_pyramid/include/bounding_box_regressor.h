#ifndef BOUNDING_BOX_REGRESSOR_H
#define BOUNDING_BOX_REGRESSOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <vector>

#include "deep_pyramid.h"

class BoundingBoxRegressor
{
public:
    void save(const std::string filename);
    void load(const std::string filename);

    void regress(std::vector<ObjectBox>& objects, const std::vector<cv::Mat>& features);
private:
    cv::Mat xWeights;
    cv::Mat yWeights;
    cv::Mat widthWeights;
    cv::Mat heightWeights;
};

#endif
