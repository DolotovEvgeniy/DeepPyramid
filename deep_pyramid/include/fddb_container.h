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
    void save(std::string fddb_file);
    void add(const std::string image_path, const std::vector<BoundingBox> boxes);
    void next(cv::Mat& img, std::vector<cv::Rect>& objects, std::vector<float>& confidence);
private:
    std::vector<std::string> imagesPath;
    std::vector<std::vector<cv::Rect> > objectsList;
    std::vector<std::vector<float> > confidenceList;
    void increaseCounter();
    void resetCounter();
    int counter;
};

#endif
