#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "feature_map.h"

class BoundingBox
{
public:
    double confidence;
    int level;
    cv::Rect norm5Box;
    cv::Rect originalImageBox;
    FeatureMap map;
    bool operator<(BoundingBox object)
    {
        return confidence<object.confidence;
    }
};

#endif
