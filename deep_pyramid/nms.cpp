#include <nms.h>

using namespace std;
using namespace cv;



bool SimilarFaceBox::operator() (FaceBox b1, FaceBox b2)
{
    return IOU(b1.originalImageBox,b2.originalImageBox)>eps;
}

vector<Rect> nms_max(vector<FaceBox> faces, double threshold)
{
    vector<Rect> detectedFace;
    while(!faces.empty())
    {
        int max=0;
        for(unsigned int i=0;i<faces.size();i++)
        {
            if(faces[i].confidence>faces[max].confidence)
            {
                max=i;
            }
        }
        detectedFace.push_back(faces[max].originalImageBox);
        vector<FaceBox> newFaces;
        for(unsigned int i=0;i<faces.size();i++)
        {
            if(IOU(faces[max].originalImageBox,faces[i].originalImageBox)<=threshold)
            {
                newFaces.push_back(faces[i]);
            }

        }
        faces=newFaces;
    }
    return detectedFace;
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

vector<Rect> nms_avg(vector<FaceBox> faces, double box_threshold, double confidence_threshold)
{
    vector<Rect> detectedFace;

    vector<int> labels;
    partition(faces,labels,SimilarFaceBox(box_threshold));
    int maxClusterNum=1;
    for(unsigned int i=0;i<labels.size();i++)
    {
        if(labels[i]+1>maxClusterNum)
        {
            maxClusterNum=labels[i]+1;
        }
    }
    for(int i=0;i<maxClusterNum;i++)
    {
        vector<FaceBox> boxInCluster;
        for(unsigned int j=0;j<labels.size();j++)
        {
            if(labels[j]==i)
            {
                boxInCluster.push_back(faces[j]);
            }
        }
        cout<<"Box in cluster:"<<boxInCluster.size()<<endl;
        int max=0;
        for(unsigned int i=0;i<boxInCluster.size();i++)
        {
            if(boxInCluster[i].confidence>boxInCluster[max].confidence)
            {
                max=i;
            }
        }
        double maxConfidence=boxInCluster[max].confidence;
        vector<Rect> rectWithMaxConfidence;
        for(unsigned int i=0;i<boxInCluster.size();i++)
        {
            if(boxInCluster[i].confidence>confidence_threshold*maxConfidence)
            {
                rectWithMaxConfidence.push_back(boxInCluster[i].originalImageBox);
            }
        }
        cout<<"box with max conf:"<<rectWithMaxConfidence.size()<<endl;
        detectedFace.push_back(avg_rect(rectWithMaxConfidence));
    }

    return detectedFace;
}
