// Copyright 2016 Dolotov Evgeniy

#ifndef NMS_H
#define NMS_H

#include "bounding_box.h"

typedef std::vector<BoundingBox> BoundingBoxCluster;

class NMS
{
public:
    void processBondingBox(std::vector<BoundingBox>& objects,const double& box_threshold,
                           const double& confidence_threshold=0);
protected:
    virtual void divideIntoClusters(std::vector<BoundingBox>& objects,const double& box_threshold,
                                    std::vector<BoundingBoxCluster>& clusters);
    virtual BoundingBox mergeCluster(BoundingBoxCluster& cluster, const double& confidence_threshold);
    virtual cv::Rect groupBoundingBox(std::vector<BoundingBox>& objects)=0;

};

class NMSweightedAvg :public NMS
{
    virtual cv::Rect groupBoundingBox(std::vector<BoundingBox>& objects);
};

#endif
