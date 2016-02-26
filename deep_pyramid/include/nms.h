#ifndef NMS_H
#define NMS_H

#include "bounding_box.h"

typedef std::vector<BoundingBox> BoundingBoxCluster;

class NMS
{
public:
    void processBondingBox(std::vector<BoundingBox>& objects,const double& box_threshold,
                           const double& confidence_threshold);
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

/*
class NMS
{
public:
    static void nms_max(std::vector<BoundingBox>& objects, double threshold);

    static void nms_avg(std::vector<BoundingBox>& objects, double box_threshold, double confidence_threshold);

    static void nms_intersect(std::vector<BoundingBox>& objects, double box_threshold, double confidence_threshold);
};
*/


#endif
