#include "rectangle_transform.h"

using namespace cv;

Point getRectangleCenter(const Rect &rect)
{
    return Point(rect.x+rect.width/2.0, rect.y+rect.height/2.0);
}

Rect makeRectangle(const Point& center, const int& width, const int& height)
{
    return Rect(center.x-width/2.0, center.y-height/2.0, width, height);
}
