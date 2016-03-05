#include "root_filter.h"
#include <math.h>
using namespace std;
using namespace cv;

void RootFilter::processFeatureMap(const FeatureMap &map, vector<Rect> &detectedRect,
                                   vector<double>& confidence)
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
                detectedRect.push_back(Rect(Point(width, height), filterSize));
                confidence.push_back(std::fabs(classify(extractedMap, true)));
            }
        }
    }
}

float RootFilter::classify(const FeatureMap &map, bool returnDFVal)
{
    Mat feature;
    map.reshapeToVector(feature);
    return svm->predict(feature, returnDFVal);
}

RootFilter::RootFilter(Size _filterSize, CvSVM *_svm, int _stride)
{
    filterSize=_filterSize;
    svm=_svm;
    stride=_stride;
}
