#ifndef NMS_H
#define NMS_H

#include <deep_pyramid.h>
class NMS
{
public:
    static void nms_max(std::vector<ObjectBox>& objects, double threshold);

    static void nms_avg(std::vector<ObjectBox>& objects, double box_threshold, double confidence_threshold);

};


#endif
