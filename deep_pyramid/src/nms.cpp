#include <nms.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <algorithm>

using namespace std;
using namespace cv;


void NMS::nms_max(vector<ObjectBox>& objects, double threshold)
{
    vector<ObjectBox> detectedObjects;
    while(!objects.empty())
    {
        ObjectBox objectWithMaxConfidence=*max_element(objects.begin(),objects.end());
        detectedObjects.push_back(objectWithMaxConfidence);
        vector<ObjectBox> newObjects;
        for(unsigned int i=0;i<objects.size();i++)
        {
            if(IOU(objectWithMaxConfidence.originalImageBox,objects[i].originalImageBox)<=threshold)
            {
                newObjects.push_back(objects[i]);
            }
        }
        objects=newObjects;
    }
    objects=detectedObjects;
}
Rect avg_rect(vector<Rect> rectangles)
{
    Rect resultRect;

    double sumOfX=0,sumOfY=0,sumOfWidth=0, sumOfHeight=0;
    for(unsigned int i=0;i<rectangles.size();i++)
    {
        sumOfX+=rectangles[i].x;
        sumOfY+=rectangles[i].y;
        sumOfWidth+=rectangles[i].width;
        sumOfHeight+=rectangles[i].height;
    }
    int n=rectangles.size();
    resultRect.x=sumOfX/n;
    resultRect.y=sumOfY/n;
    resultRect.width=sumOfWidth/n;
    resultRect.height=sumOfHeight/n;
    return resultRect;
}

void NMS::nms_avg(vector<ObjectBox>& objects, double box_threshold, double confidence_threshold)
{
    vector<ObjectBox> detectedObjects;
    vector< vector<ObjectBox> > clusters;
    vector<double> maxConfidenceInCluster;
    while(!objects.empty())
    {
        ObjectBox objectWithMaxConfidence=*max_element(objects.begin(),objects.end());
        vector<ObjectBox> cluster;
       // cluster.push_back(objectWithMaxConfidence);
        vector<ObjectBox> newObjects;
        for(unsigned int i=0;i<objects.size();i++)
        {
            if(IOU(objectWithMaxConfidence.originalImageBox,objects[i].originalImageBox)<=box_threshold)
            {
                newObjects.push_back(objects[i]);
            }
            else
            {
                cluster.push_back(objects[i]);
            }

        }
        clusters.push_back(cluster);
        maxConfidenceInCluster.push_back(objectWithMaxConfidence.confidence);
        objects=newObjects;
    }
    for(unsigned int clusterNum=0;clusterNum<clusters.size();clusterNum++)
    {
        vector<ObjectBox> boxInCluster;
        boxInCluster=clusters[clusterNum];
        cout<<"Box in cluster:"<<boxInCluster.size()<<endl;
        vector<Rect> rectWithMaxConfidence;
        for(unsigned int j=0;j<boxInCluster.size();j++)
        {
            if(boxInCluster[j].confidence>confidence_threshold*maxConfidenceInCluster[clusterNum])
            {
                rectWithMaxConfidence.push_back(boxInCluster[j].originalImageBox);
            }
        }
        cout<<"box with max conf:"<<rectWithMaxConfidence.size()<<endl;
        ObjectBox resultObject;
        resultObject.originalImageBox=avg_rect(rectWithMaxConfidence);
        resultObject.confidence=maxConfidenceInCluster[clusterNum];
        detectedObjects.push_back(resultObject);
    }

    objects=detectedObjects;
}
