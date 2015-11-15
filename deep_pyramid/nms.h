#ifndef NMS_H
#define NMS_H
#include <deep_pyramid.h>
std::vector<ObjectBox> nms_max(std::vector<ObjectBox> objects, double threshold);

std::vector<ObjectBox> nms_avg(std::vector<ObjectBox> objects, double box_threshold, double confidence_threshold);

#endif // NMS_H
