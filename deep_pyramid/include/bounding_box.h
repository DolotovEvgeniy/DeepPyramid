#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

enum BoxLevel{NORM5, ORIGINAL};

class BoundingBox
{
public:
    double confidence;
    int level;
    cv::Rect norm5Box;
    cv::Rect originalImageBox;
    bool operator<(BoundingBox object)
    {
        return confidence<object.confidence;
    }
};

double IOU(const cv::Rect& rect1, const cv::Rect& rect2);

#endif
