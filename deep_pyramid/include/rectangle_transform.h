#ifndef RECTANGLE_TRANSFORM_H
#define RECTANGLE_TRANSFORM_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

cv::Point getRectangleCenter(const cv::Rect& rect);

cv::Rect makeRectangle(const cv::Point& center, const int& width, const int& height);

cv::Rect avg_rect(const std::vector<cv::Rect>& rectangles);

cv::Rect intersectRectangles(const std::vector<cv::Rect>& rectangles);

cv::Rect scaleRect(cv::Rect rect, double scale);

double IOU(const cv::Rect& rect1, const cv::Rect& rect2);

#endif
