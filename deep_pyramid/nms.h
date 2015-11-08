#ifndef NMS_H
#define NMS_H
#include <deep_pyramid.h>
std::vector<cv::Rect> nms_max(std::vector<FaceBox> faces, double threshold);

class SimilarFaceBox
{
public:
    double eps;
    SimilarFaceBox(double _eps):eps(_eps){}
    bool operator() (FaceBox b1, FaceBox b2);
};

std::vector<cv::Rect> nms_avg(std::vector<FaceBox> faces, double box_threshold, double confidence_threshold);


#endif // NMS_H
