#ifndef NMS_H
#define NMS_H

#include "bounding_box.h"

class NMS
{
public:
    static void nms_max(std::vector<BoundingBox>& objects, double threshold);

    static void nms_avg(std::vector<BoundingBox>& objects, double box_threshold, double confidence_threshold);

    static void nms_intersect(std::vector<BoundingBox>& objects, double box_threshold, double confidence_threshold);
};



#endif
