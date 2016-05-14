// Copyright 2016 Dolotov Evgeniy

#ifndef FDDB_CONTAINER_H
#define FDDB_CONTAINER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <utility>
#include <string>

#include <bounding_box.h>

class FDDBContainer
{
public:
    FDDBContainer() {}
    void load(std::string fddb_file, std::string image_prefix="");
    void next(std::string& image_path, cv::Mat& img, std::vector<cv::Rect>& objects);
    int size();
    int objectsCount();
    void reset();
private:
    std::vector<std::string> imagesPath;
    std::vector<std::vector<cv::Rect> > objectsList;
    void increaseCounter();
    void resetCounter();
    int counter;
    std::string prefix;
};

#endif
