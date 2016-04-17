// Copyright 2016 Dolotov Evgeniy

#include "../include/rectangle_transform.h"

#include <vector>

using cv::Rect;
using cv::Point;
using std::vector;

Point getRectangleCenter(const Rect &rect) {
    return Point(rect.x+rect.width/2.0, rect.y+rect.height/2.0);
}

Rect makeRectangle(const Point& center, const int& width, const int& height) {
    CV_Assert(width > 0 && height > 0);
    return Rect(center.x-width/2.0, center.y-height/2.0, width, height);
}

double IOU(const Rect& rect1, const Rect& rect2) {
    Rect unionRect = rect1 & rect2;

    return unionRect.area()/(double)(rect1.area()+rect2.area()-unionRect.area());
}

Rect avg_rect(const vector<Rect>& rectangles) {
    CV_Assert(rectangles.size() > 0);

    Rect resultRect;

    double sumOfX = 0, sumOfY = 0, sumOfWidth = 0, sumOfHeight = 0;
    for (unsigned int i = 0; i < rectangles.size(); i++) {
        sumOfX+=rectangles[i].x;
        sumOfY+=rectangles[i].y;
        sumOfWidth+=rectangles[i].width;
        sumOfHeight+=rectangles[i].height;
    }
    int n = rectangles.size();
    resultRect.x = sumOfX/n;
    resultRect.y = sumOfY/n;
    resultRect.width = sumOfWidth/n;
    resultRect.height = sumOfHeight/n;
    return resultRect;
}

Rect intersectRectangles(const vector<Rect>& rectangles) {
    Rect resultRectangle = rectangles[0];

    for (vector<Rect>::const_iterator rectangle = rectangles.begin(); rectangle != rectangles.end(); rectangle++) {
        resultRectangle = resultRectangle & (*rectangle);
    }

    return resultRectangle;
}

Rect scaleRect(Rect rect, double scale) {
    CV_Assert(scale > 0);
    rect.x*=scale;
    rect.y*=scale;
    rect.width*=scale;
    rect.height*=scale;
    return rect;
}
