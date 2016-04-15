#include <feature_map_svm.h>

using namespace std;
using namespace cv;

void FeatureMapSVM::load(const string& filename)
{
    svm=new CvSVM();
    svm->load(filename.c_str());
}

void FeatureMapSVM::save(const string &filename)
{
    svm->save(filename.c_str());
}

float FeatureMapSVM::predict(const FeatureMap &sample, bool returnDFVal) const
{
    Mat feature;
    sample.reshapeToVector(feature);

    return svm->predict(feature, returnDFVal);
}

void FeatureMapSVM::train(const vector<FeatureMap> &positive, const vector<FeatureMap>& negative)
{
    Mat features;
    Mat labels;

    for(unsigned int i=0;i<positive.size();i++)
    {
        Mat feature;

        positive[i].reshapeToVector(feature);
        features.push_back(feature);
        labels.push_back(OBJECT);
    }

    for(unsigned int i=0;i<negative.size();i++)
    {
        Mat feature;

        negative[i].reshapeToVector(feature);
        features.push_back(feature);
        labels.push_back(NOT_OBJECT);
    }

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    svm->train_auto(features,labels, Mat(),Mat(), params);
}

FeatureMapSVM::~FeatureMapSVM()
{
   delete svm;
}

void FeatureMapSVM::printAccuracy(const vector<FeatureMap> &positive, const vector<FeatureMap>& negative)
{
    int objectsCount=positive.size();
    int negativeCount=negative.size();

    int trueNegative=0;
    int truePositive=0;

    for(unsigned int i=0;i<positive.size();i++)
    {
        if(predict(positive[i])==OBJECT)
        {
            truePositive++;
        }
    }
    for(unsigned int i=0;i<negative.size();i++)
    {
        if(predict(negative[i])==OBJECT)
        {
            trueNegative++;
        }
    }

    cout<<"Objects classification accuracy:"<<truePositive/objectsCount<<endl;
    cout<<"Negative classification accuracy:"<<trueNegative/negativeCount<<endl;
    cout<<"Common classification accuracy:"<<(truePositive+trueNegative)/(objectsCount+negativeCount);

    cout<<"Count of features:"<<objectsCount+negativeCount<<endl;
}
