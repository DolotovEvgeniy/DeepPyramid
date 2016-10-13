// Copyright 2016 Dolotov Evgeniy

#ifndef DEEP_PYRAMID_INCLUDE_DETECT_RESULT_CONTAINER_H_
#define DEEP_PYRAMID_INCLUDE_DETECT_RESULT_CONTAINER_H_

#include <vector>
#include <utility>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "bounding_box.h"

class DetectResultContainer {
private:
    std::vector<std::string> imagesPath;
    std::vector<std::vector<cv::Rect> > objectsList;
    std::vector<std::vector<float> > confidenceList;
public:
    DetectResultContainer() {}
    void save(std::string file_name);
    void add(std::string image_path, std::vector<cv::Rect> objects, std::vector<float> confidence);
    int size();
    int detectedObjectsCount();
};

#endif  // DEEP_PYRAMID_INCLUDE_DETECT_RESULT_CONTAINER_H_
