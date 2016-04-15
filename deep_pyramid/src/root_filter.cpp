#include "root_filter.h"
#include <math.h>
using namespace std;
using namespace cv;

void RootFilter::processFeatureMap(const FeatureMap &map, vector<BoundingBox> &detectedObjects, int stride) const
{
    Size mapSize=map.size();
    for(int width=0; width<mapSize.width-filterSize.width; width+=stride)
    {
        for(int height=0; height<mapSize.height-filterSize.height; height+=stride)
        {
            FeatureMap extractedMap;
            map.extractFeatureMap(Rect(Point(width, height), filterSize), extractedMap);
            if(((int)classify(extractedMap))==OBJECT)
            {
                BoundingBox box;
                box.norm5Box=Rect(Point(width, height), filterSize);
                box.confidence=std::fabs(classify(extractedMap, true));
                box.map=extractedMap;
                detectedObjects.push_back(box);
            }
        }
    }
}

float RootFilter::classify(const FeatureMap &map, bool returnDFVal) const
{
    return svm->predict(map, returnDFVal);
}

RootFilter::RootFilter(Size _filterSize, string svm_file)
{
    filterSize=_filterSize;
    svm=new FeatureMapSVM;
    svm->load(svm_file);
    cout<<"here&&&&&&"<<endl;
}

Size RootFilter::getFilterSize()
{
    return filterSize;
}
