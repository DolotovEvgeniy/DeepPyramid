#include <deep_pyramid.h>
#include <nms.h>
#include <assert.h>
#include <time.h>
#include <string>
#include <algorithm>
using namespace cv;
using namespace std;
using namespace caffe;

Rect scaleRect(Rect r, double scale)
{
    r.x*=scale;
    r.y*=scale;
    r.width*=scale;
    r.height*=scale;
    return r;
}

Rect DeepPyramid::getRectByNorm5Pixel_ARTICLE(Point p)
{
    Point center=p*centerConformity;
    //high-left
    Point hl;
    int x,y;
    x=center.x-boxSideConformity;
    y=center.y-boxSideConformity;
    hl.x=x>0 ? x : 0;
    hl.y=y>0 ? y : 0;

    Blob<float>* input_layer = net->input_blobs()[0];
    int networkInputSize=input_layer->width();
    //down-right
    Point dr;
    x=center.x+boxSideConformity;
    y=center.y+boxSideConformity;
    dr.x=x<networkInputSize ? x : networkInputSize-1;
    dr.y=y<networkInputSize ? y : networkInputSize-1;

    return Rect(hl,dr);
}

Rect DeepPyramid::getRectByNorm5Rect_ARTICLE(Rect r)
{
    Point hl=Point(r.x,r.y);
    Point dr=Point(r.x+r.width,r.y+r.height);
    Rect r1=getRectByNorm5Pixel_ARTICLE(hl);
    Rect r2=getRectByNorm5Pixel_ARTICLE(dr);
    Point RectHL=Point(r1.x,r1.y);
    Point RectDR=Point(r2.x+r2.width, r2.y+r2.height);

    return Rect(RectHL, RectDR);
}

Rect DeepPyramid::getNorm5RectByOriginal_ARTICLE(Rect originalRect)
{
    Point hl=Point(originalRect.x,originalRect.y);
    Point dr=Point(originalRect.x+originalRect.width,originalRect.y+originalRect.height);
    Point norm5HL=(hl+Point(boxSideConformity,boxSideConformity))*(1/(double)centerConformity);
    Point norm5DR=(dr-Point(boxSideConformity,boxSideConformity))*(1/(double)centerConformity);
    Rect r;
    if(norm5HL.x<norm5DR.x)
    {
        r.x=norm5HL.x;
        r.width=norm5DR.x-norm5HL.x;
    }
    else
    {
        r.x=(rand()%2)==0 ? norm5DR.x : norm5HL.x;
        r.width=1;
    }
    if(norm5HL.y<norm5DR.y)
    {
        r.y=norm5HL.y;
        r.height=norm5DR.y-norm5HL.y;
    }
    else
    {
        r.y=(rand()%2)==0 ? norm5DR.y : norm5HL.y;
        r.height=1;
    }
    return r;
}

Rect DeepPyramid::originalRect2Norm5(const Rect& originalRect, int level, const Size& imgSize)
{
    double longSide=std::max(imgSize.height, imgSize.width);
    Blob<float>* input_layer = net->input_blobs()[0];
    int networkInputSize=input_layer->width();
    double scalePyramid = networkInputSize/(pow(2, (config.numLevels-1-level)/2.0)*longSide);

    Rect boundBoxInPyramid;
    boundBoxInPyramid=scaleRect(originalRect,scalePyramid);

    Blob<float>* output_layer = net->output_blobs()[0];
    int networkOutputSize=output_layer->width();
    double scaleNorm5=networkInputSize/(double)networkOutputSize;

    Rect boundBoxInNorm5;
    boundBoxInNorm5=scaleRect(boundBoxInPyramid, scaleNorm5);

    return boundBoxInNorm5;
}

Rect DeepPyramid::norm5Rect2Original(const Rect& norm5Rect, int level, const Size& imgSize)
{
    Blob<float>* input_layer = net->input_blobs()[0];
    Blob<float>* output_layer = net->output_blobs()[0];
    int networkInputSize=input_layer->width();
    int networkOutputSize=output_layer->width();
    double scaleNorm5=networkInputSize/(double)networkOutputSize;

    Rect boundBoxInPyramid;
    boundBoxInPyramid=scaleRect(norm5Rect,scaleNorm5);

    double longSide=std::max(imgSize.height, imgSize.width);
    double scalePyramid = (longSide*pow(2, (config.numLevels-1-level)/2.0))/networkInputSize;
    Rect boundBoxOriginal;
    boundBoxOriginal=scaleRect(boundBoxInPyramid, scalePyramid);

    return boundBoxInPyramid;
}

//Image Pyramid
//
Size DeepPyramid::imageSizeAtPyramidLevel(const Mat& img, const int& i)
{
    Size levelPyramidImageSize(img.cols, img.rows);

    Blob<float>* input_layer = net->input_blobs()[0];
    int networkInputSize=input_layer->width();

    if(img.rows<=img.cols)
    {
        levelPyramidImageSize.width=networkInputSize/pow(2,(config.numLevels-1-i)/2.0);
        levelPyramidImageSize.height=levelPyramidImageSize.width*(img.rows/((double)img.cols));
    }
    else
    {
        levelPyramidImageSize.height=networkInputSize/pow(2,(config.numLevels-1-i)/2.0);
        levelPyramidImageSize.width=levelPyramidImageSize.height*(img.cols/((double)img.rows));
    }

    return levelPyramidImageSize;
}

void DeepPyramid::createImageAtPyramidLevel(const Mat& img, const int& i, Mat& dst)
{
    Blob<float>* input_layer = net->input_blobs()[0];
    int networkInputSize=input_layer->width();

    dst=Mat(networkInputSize, networkInputSize, CV_8UC3, Scalar::all(0));
    Size pictureSize=imageSizeAtPyramidLevel(img, i);

    Mat resizedImg;
    resize(img, resizedImg, pictureSize);
    resizedImg.copyTo(dst(Rect(Point(0,0),pictureSize)));
}

void DeepPyramid::createImagePyramid(const Mat& img)
{
    imagePyramid.clear();
    for(int i=0; i<config.numLevels; i++)
    {
        Mat imgAtLevel;
        createImageAtPyramidLevel(img, i, imgAtLevel);
        imagePyramid.push_back(imgAtLevel);
    }
    cout<<"Create image pyramid..."<<endl;
    cout<<"Status: Success!"<<endl;
}
//
////

//NeuralNet
//
void DeepPyramid::calculateImageRepresentation()
{
    net->ForwardPrefilled();
}

void DeepPyramid::fillNeuralNetInput(int i)
{
    Mat img=imagePyramid[i];

    Blob<float>* input_layer = net->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    input_layer->Reshape(1, input_layer->channels(), height, width);
    net->Reshape();

    std::vector<Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    Mat img_float;
    img.convertTo(img_float, CV_32FC3);
    split(img_float, input_channels); // save
}

void DeepPyramid::calculateImageRepresentationAtLevel(const int& i)
{
    const clock_t begin_time = clock();
    cout<<"Calculate level: "+to_string(i+1)+" ..."<<endl;
    fillNeuralNetInput(i);
    calculateImageRepresentation();
    cout << "Time:"<<float( clock () - begin_time ) /  CLOCKS_PER_SEC<<" s."<<endl;
    cout<<"Status: Success!"<<endl;

}

void DeepPyramid::getNeuralNetOutput(vector<Mat>& netOutput)
{
    Blob<float>* output_layer = net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    float* data=new float[output_layer->height()*output_layer->width()];

    for(int k=0;k<output_layer->channels();k++)
    {
        for(int i=0;i<output_layer->height()*output_layer->width();i++)
        {
            data[i]=begin[i+output_layer->height()*output_layer->width()*k];
        }
        Mat conv(output_layer->height(),output_layer->width(), CV_32FC1, data);
        netOutput.push_back(conv.clone());
    }
}
//
////

//Max5
//
void DeepPyramid::createMax5AtLevel(const int& i, vector<Mat>& max5)
{
    calculateImageRepresentationAtLevel(i);
    getNeuralNetOutput(max5);
}
void DeepPyramid::createMax5Pyramid()
{
    for(int i=0; i<config.numLevels; i++)
    {
        vector<Mat> max5AtLevel;
        createMax5AtLevel(i, max5AtLevel);
        max5.push_back(max5AtLevel);
    }
}
//
////

//Norm5
//
Mat uniteMats(std::vector<Mat> m)
{
    Mat unite(1,0,CV_32FC1);
    for(unsigned int i=0;i<m.size();i++)
    {
        unite.push_back(m[i].reshape(1,1));
    }

    return unite;
}

void calculateMeanAndDeviationValue(std::vector<Mat> level, float& meanValue, float& deviationValue)
{
    Mat unite=uniteMats(level);
    Mat mean, deviation;
    //корень или квадрат?
    meanStdDev(unite, mean, deviation);
    meanValue=mean.at<double>(0,0);
    deviationValue=deviation.at<double>(0,0);
}

void DeepPyramid::createNorm5AtLevel(const int& i, vector<Mat>& norm5)
{
    float mean,deviation;
    calculateMeanAndDeviationValue(max5[i],mean,deviation);
    for(unsigned int j=0;j<max5[i].size();j++)
    {
        norm5.push_back((max5[i][j]-mean)/deviation);
    }
}

void DeepPyramid::createNorm5Pyramid()
{
    for(unsigned int i=0;i<max5.size();i++)
    {
        vector<Mat> norm5AtLevel;
        createNorm5AtLevel(i, norm5AtLevel);
        norm5.push_back(norm5AtLevel);
    }
}
//
////

//Root-Filter sliding window
//
void DeepPyramid::getNegFeatureVector(int levelIndx, const Rect& rect, Mat& feature)
{
    for(unsigned int k=0;k<norm5[levelIndx].size();k++)
    {
        Mat m;
        norm5[levelIndx][k](rect).copyTo(m);
        for(int h=0;h<rect.height;h++)
        {
            for(int w=0;w<rect.width;w++)
            {
                feature.at<float>(0,w+h*rect.width+k*rect.area())=m.at<float>(h, w);
            }
        }
    }
}

void DeepPyramid::getPosFeatureVector(const Rect& rect, const Size& size, Mat& feature, const Size& imgSize)
{
    cout<<"rect"<<rect<<endl;
    int bestLevel=chooseLevel(size, rect, imgSize);
    cout<<"level:"<<bestLevel<<endl;
    Rect objectRect=originalRect2Norm5(rect, bestLevel, imgSize);
    cout<<"objectRect:"<<objectRect<<endl;
    vector<Mat> norm5Resized;
    for(int j=0;j<256;j++)
    {
        Mat m;
        Mat objectMat=norm5[bestLevel][j];
        resize(objectMat(objectRect), m, size);
        norm5Resized.push_back(m);
    }

    for(int w=0;w<size.width;w++)
    {
        for(int h=0;h<size.height;h++)
        {
            for(unsigned int k=0;k<norm5.size();k++)
            {
                int featureIndex=w+h*size.width+k*size.height*size.width;
                feature.at<float>(0,featureIndex)=norm5Resized[k].at<float>(h,w);
            }
        }
    }
}

void DeepPyramid::rootFilterAtLevel(int rootFilterIndx, int levelIndx, vector<ObjectBox>& detectedObjects)
{
    Size filterSize=rootFilter[rootFilterIndx].first;
    CvSVM* filterSVM=rootFilter[rootFilterIndx].second;
    int stepWidth, stepHeight;

    int stride=config.stride;
    stepWidth=((norm5[levelIndx][0].cols/pow(2,(config.numLevels-1-levelIndx)/2.0)-filterSize.width)/stride)+1;
    stepHeight=((norm5[levelIndx][0].rows/pow(2,(config.numLevels-1-levelIndx)/2.0)-filterSize.height)/stride)+1;
    int detectedObjectCount=0;

    for(int w=0;w<stepWidth;w+=stride)
        for(int h=0;h<stepHeight;h+=stride)
        {

            Point p(stride*w,stride*h);
            cv::Mat feature(1,filterSize.height*filterSize.width*norm5[levelIndx].size(),CV_32FC1);
            getNegFeatureVector(levelIndx, Rect(p, filterSize), feature);

            int predict;
            predict=filterSVM->predict(feature);

            if(predict==OBJECT)
            {
                ObjectBox object;
                object.confidence=std::fabs(filterSVM->predict(feature,true));
                object.level=levelIndx;
                object.norm5Box=Rect(p,filterSize);
                detectedObjects.push_back(object);
                detectedObjectCount++;
            }
        }
    cout<<"Count of detected objects: "<<detectedObjectCount<<endl;
}

void DeepPyramid::rootFilterConvolution(vector<ObjectBox>& detectedObjects)
{
    for(unsigned int i=0;i<rootFilter.size();i++)
        for(unsigned int j=0;j<norm5.size();j++)
        {
            cout<<"Convolution with SVM level "+to_string(j+1)+"..."<<endl;
            const clock_t begin_time = clock();
            rootFilterAtLevel(i, j, detectedObjects);
            cout << "Time:"<<float( clock () - begin_time ) /  CLOCKS_PER_SEC<<" s."<<endl;
            cout<<"Status: Success!"<<endl;
        }
}
//
////

//Rectangle
//

void DeepPyramid::calculateOriginalRectangle(vector<ObjectBox>& detectedObjects, const Size& imgSize)
{
    for(unsigned int i=0;i<detectedObjects.size();i++)
    {
        Rect originalRect=norm5Rect2Original(detectedObjects[i].norm5Box, detectedObjects[i].level, imgSize);
        detectedObjects[i].originalImageBox=originalRect;
    }
}
double IOU(const Rect& r1, const Rect& r2)
{
    Rect runion= r1 & r2;

    return (double)runion.area()/(r1.area()+r2.area()-runion.area());
}

void DeepPyramid::groupOriginalRectangle(vector<ObjectBox>& detectedObjects)
{
    NMS::nms_avg(detectedObjects,0.2,0.7);
}

void DeepPyramid::detect(const Mat& img, vector<ObjectBox>& objects)
{
    assert(img.channels()==3);

    clear();
    createImagePyramid(img);
    createMax5Pyramid();
    createNorm5Pyramid();
    cout<<"filter"<<endl;
    vector<ObjectBox> detectedObjects;
    rootFilterConvolution(detectedObjects);
    cout<<"group rectangle"<<endl;
    calculateOriginalRectangle(detectedObjects, Size(img.cols, img.rows));
    groupOriginalRectangle(detectedObjects);
    objects=detectedObjects;
    cout<<"boundbox regressor: TODO"<<endl;
    cout<<"Object count:"<<detectedObjects.size()<<endl;
}

void DeepPyramid::clear()
{
    imagePyramid.clear();
    max5.clear();
    norm5.clear();
}

void DeepPyramid::calculateToNorm5(const Mat& img)
{
    clear();
    createImagePyramid(img);
    createMax5Pyramid();
    createNorm5Pyramid();
}

int DeepPyramid::chooseLevel(const Size& filterSize,const Rect& boundBox, const Size& imgSize)
{
    vector<double> f;
    for(int i=0;i<config.numLevels;i++)
    {
        Rect r=originalRect2Norm5(boundBox, i, imgSize);

        f.push_back(abs(filterSize.width-r.width)+abs(r.height-filterSize.height));
    }
    int bestLevel=distance(f.begin(), min_element(f.begin(), f.end()));

    return bestLevel;
}

DeepPyramid::DeepPyramid(FileStorage& configFile) : config(configFile)
{
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    net.reset(new Net<float>(config.model_file, caffe::TEST));
    net->CopyTrainedLayersFrom(config.trained_net_file);
    Blob<float>* input_layer = net->input_blobs()[0];
    assert(input_layer->width()==input_layer->height());

    centerConformity=16;
    boxSideConformity=81;
    CvSVM* classifier=new CvSVM();
    classifier->load(config.svm_trained_file.c_str());
    rootFilter.push_back(std::make_pair(config.filterSize, classifier));
}


DeepPyramidConfiguration::DeepPyramidConfiguration(FileStorage& configFile)
{
    configFile["NeuralNetwork-configuration"]>>model_file;
    configFile["NeuralNetwork-trained-model"]>>trained_net_file;
    configFile["NumberOfLevel"]>>numLevels;

    configFile["Stride"]>>stride;
    configFile["ObjectColor"]>>objectRectangleColor;

    configFile["SVM"]>>svm_trained_file;
    configFile["Filter-size"]>>filterSize;
}

bool isContain(const Mat& img, Rect rect)
{
    return (rect.x>0 && rect.y>0 && rect.x+rect.width>img.cols && rect.y+rect.height>img.rows);
}

void DeepPyramid::extractFeatureVectors(const Mat& img, const int& filterIdx, const vector<Rect>& objectsRect, Mat& features, Mat& labels)
{
    calculateToNorm5(img);
    cout<<"Cut negatives"<<endl;

    for(int level=0; level<config.numLevels; level++)
    {
        vector<Rect> norm5Objects;
        for(unsigned int j=0; j<norm5Objects.size(); j++)
        {
            Rect objectRect=originalRect2Norm5(objectsRect[j], level, Size(img.cols, img.rows));
            norm5Objects.push_back(objectRect);
        }

        int stepWidth, stepHeight, stride=config.stride;

        stepWidth=((norm5[0][0].cols/pow(2,(config.numLevels-1-level)/2.0)-rootFilter[filterIdx].first.width)/stride)+1;
        stepHeight=((norm5[0][0].rows/pow(2,(config.numLevels-1-level)/2.0)-rootFilter[filterIdx].first.height)/stride)+1;

        for(int w=0;w<stepWidth;w+=stride)
            for(int h=0;h<stepHeight;h+=stride)
            {
                Point p(stride*w, stride*h);
                Rect sample(p, rootFilter[filterIdx].first);

                bool addNeg=true;
                for(unsigned int objectNum=0; objectNum<norm5Objects.size() ;objectNum++)
                {
                    if(IOU(norm5Objects[objectNum],sample)>0.3)
                    {
                        addNeg=false;
                        break;
                    }
                }

                if(addNeg)
                {
                    Mat feature(1,rootFilter[filterIdx].first.area()*norm5[0].size(),CV_32FC1);
                    getNegFeatureVector(level, sample, feature);
                    features.push_back(feature);
                    labels.push_back(NOT_OBJECT);
                }

            }
    }
    cout<<"Cut positive"<<endl;
    for(unsigned int i=0;i<objectsRect.size();i++)
    {
        if(isContain(img, objectsRect[i]))
        {
            Mat feature(1,rootFilter[filterIdx].first.area()*norm5[0].size(),CV_32FC1);
            getPosFeatureVector(objectsRect[i], rootFilter[filterIdx].first, feature, Size(img.cols, img.rows));

            features.push_back(feature);
            labels.push_back(OBJECT);
        }
    }

}

DeepPyramid::~DeepPyramid()
{
    for(unsigned int i=0;i<rootFilter.size();i++)
    {
        delete rootFilter[i].second;
    }
}
