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
    virtual BoundingBox mergeCluster(BoundingBoxCluster& cluster, const double& confidence_threshold)=0;
private:
    void divideIntoClusters(std::vector<BoundingBox>& objects,const double& box_threshold,
                            std::vector<BoundingBoxCluster>& clusters);
};

class NMSmax : public NMS
{
    BoundingBox mergeCluster(BoundingBoxCluster& cluster, const double& confidence_threshold);
};

class NMSavg : public NMS
{
    BoundingBox mergeCluster(BoundingBoxCluster& cluster, const double& confidence_threshold);
};

class NMSintersect : public NMS
{
    BoundingBox mergeCluster(BoundingBoxCluster& cluster, const double& confidence_threshold);
};

class NMSweightedAvg :public NMS
{
    BoundingBox mergeCluster(BoundingBoxCluster& cluster, const double& confidence_threshold);
};

#endif
