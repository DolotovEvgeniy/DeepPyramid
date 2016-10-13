// Copyright 2016 Dolotov Evgeniy

#include <vector>

#include "rectangle_transform.h"

using namespace std;
using namespace cv;

Point getRectangleCenter(const Rect& rect) {
    return Point(rect.x+rect.width/2.0, rect.y+rect.height/2.0);
}

Rect makeRectangle(const Point& center, const int& width, const int& height) {
    CV_Assert(width > 0 && height > 0);
    return Rect(center.x-width/2.0, center.y-height/2.0, width, height);
}

double IOU(const Rect& rect1, const Rect& rect2) {
    Rect intersectRect = rect1 & rect2;

    int unionRectArea = rect1.area()+rect2.area()-intersectRect.area();

    return intersectRect.area()/(double)unionRectArea;
}

Rect weightedAvg_rect(const vector<BoundingBox>& rectangles)
{
    CV_Assert(rectangles.size() > 0);

    Rect resultRect;

    double weight = 0.0;
    for (size_t i = 0; i < rectangles.size(); i++) {
        weight += rectangles[i].confidence;
    }

    double sumOfX = 0.0, sumOfY = 0.0;
    double sumOfWidth = 0.0, sumOfHeight = 0.0;
    for (size_t i = 0; i < rectangles.size(); i++) {
        Rect rectangle = rectangles[i].originalImageBox;
        double c = rectangles[i].confidence;
        double d=c/weight;
        sumOfX+=rectangle.x*d;
        sumOfY+=rectangle.y*d;
        sumOfWidth+=rectangle.width*d;
        sumOfHeight+=rectangle.height*d;
    }

    resultRect.x = sumOfX;
    resultRect.y = sumOfY;
    resultRect.width = sumOfWidth;
    resultRect.height = sumOfHeight;
    return resultRect;
}

Rect scaleRect(const Rect& rect, double scale) {
    CV_Assert(scale > 0.0);
    Rect newRect;
    newRect.x=rect.x*scale;
    newRect.y=rect.y*scale;
    newRect.width=rect.width*scale;
    newRect.height=rect.height*scale;
    return newRect;
}
