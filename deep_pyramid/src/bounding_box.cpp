#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "bounding_box.h"


using namespace cv;

double IOU(const Rect& rect1, const Rect& rect2)
{
    Rect unionRect= rect1 & rect2;

    return unionRect.area()/(double)(rect1.area()+rect2.area()-unionRect.area());
}
