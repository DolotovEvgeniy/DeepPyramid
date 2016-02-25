#ifndef RECTANGLE_TRANSFORM_H
#define RECTANGLE_TRANSFORM_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Point getRectangleCenter(const cv::Rect& rect);

cv::Rect makeRectangle(const cv::Point& center, const int& width, const int& height);

#endif
