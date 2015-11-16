#include <nms.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <algorithm>

using namespace std;
using namespace cv;


vector<ObjectBox> nms_max(vector<ObjectBox> objects, double threshold)
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
    return detectedObjects;
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

vector<ObjectBox> nms_avg(vector<ObjectBox> objects, double box_threshold, double confidence_threshold)
{
    vector<ObjectBox> detectedObjects;
    vector< vector<ObjectBox> > clusters;
    vector<double> maxConfidenceInCluster;
    while(!objects.empty())
    {
        ObjectBox objectWithMaxConfidence=*max_element(objects.begin(),objects.end());
        vector<ObjectBox> cluster;
        cluster.push_back(objectWithMaxConfidence);
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
    for(unsigned int i=0;i<clusters.size();i++)
    {
        vector<ObjectBox> boxInCluster;
        boxInCluster=clusters[i];
        cout<<"Box in cluster:"<<boxInCluster.size()<<endl;
        vector<Rect> rectWithMaxConfidence;
        for(unsigned int i=0;i<boxInCluster.size();i++)
        {
            if(boxInCluster[i].confidence>confidence_threshold*maxConfidenceInCluster[i])
            {
                rectWithMaxConfidence.push_back(boxInCluster[i].originalImageBox);
            }
        }
        cout<<"box with max conf:"<<rectWithMaxConfidence.size()<<endl;
        ObjectBox resultObject;
        resultObject.originalImageBox=avg_rect(rectWithMaxConfidence);
        resultObject.confidence=maxConfidenceInCluster[i];
        detectedObjects.push_back(resultObject);
    }

    return detectedObjects;
}
