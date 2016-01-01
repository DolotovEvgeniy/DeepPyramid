#ifndef BOUNDING_BOX_REGRESSOR_H
#define BOUNDING_BOX_REGRESSOR_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "bounding_box.h"

class BoundingBoxRegressor
{
public:
    void save(const std::string filename);
    void load(const std::string filename);

    void regress(std::vector<BoundingBox>& objects, const std::vector<cv::Mat>& features);
private:

    void regressBox(BoundingBox& object,const cv::Mat& feature);
    cv::Mat xWeights;
    cv::Mat yWeights;
    cv::Mat widthWeights;
    cv::Mat heightWeights;
};

#endif
