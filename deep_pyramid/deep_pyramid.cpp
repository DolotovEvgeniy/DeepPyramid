#include <deep_pyramid.h>

using namespace cv;
using namespace std;
using namespace caffe;

cv::Rect getRectByNorm5Pixel(cv::Point p)
{
    Point center=p*16;
    Point hl=Point((center.x-81)>0 ? center.x-81 : 0, (center.y-81)>0 ? center.y-81 : 0);
    Point dr=Point((center.x+81)<1713 ? center.x+81 : 1712, (center.y+81)<1713 ? center.y+81 : 1712);

    return Rect(hl,dr);
}
cv::Rect getRectByNorm5Rect(cv::Rect r)
{
    Point hl=Point(r.x,r.y);
    Point dr=Point(r.x+r.width,r.y+r.height);
    Rect r1=getRectByNorm5Pixel(hl);
    Rect r2=getRectByNorm5Pixel(dr);
    Point RectHL=Point(r1.x,r1.y);
    Point RectDR=Point(r2.x+r2.width, r2.y+r2.height);

    return Rect(RectHL, RectDR);
}
cv::Rect getNorm5RectByOriginal(cv::Rect originalRect)
{
    Point hl=Point(originalRect.x,originalRect.y);
    Point dr=Point(originalRect.x+originalRect.width,originalRect.y+originalRect.height);
    Point norm5HL=(hl+Point(81,81))*(1/16.0);
    Point norm5DR=(dr-Point(81,81))*(1/16.0);
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
cv::Rect boundingBoxAtLevel(int i, cv::Rect originalRect,cv::Size originalMatSize)
{
    double longSide=(originalMatSize.width>originalMatSize.height) ? originalMatSize.width : originalMatSize.height;
    double scale = (1713.0/pow(2, (6-i)/2.0))/longSide;
    originalRect.x*=scale;
    originalRect.y*=scale;
    originalRect.height*=scale;
    originalRect.width*=scale;
    return originalRect;
}



std::string to_string(int i)
{
    std::ostringstream stm ;
    stm << i ;
    return stm.str() ;
}



DeepPyramid:: DeepPyramid(int num_levels, const string& model_file,
                          const string& trained_file) {
    this->num_levels=num_levels;
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

void DeepPyramid::createMax5PyramidTest()
{
    createMax5Pyramid();
}
void DeepPyramid::showNorm5Pyramid()
{
    for(unsigned int i=0;i<norm5.size();i++)
        for(unsigned int j=0;j<norm5[i].size();j++)
        {
            imshow("norm5| "+to_string(i)+" | "+to_string(j), norm5[i][j]);
            waitKey(0);
            destroyWindow("norm5| "+to_string(i)+" | "+to_string(j));
        }
}

void DeepPyramid::setImg(Mat img)
{
    img.copyTo(originalImg);
}

void DeepPyramid::showImagePyramid()
{
    for(int i=0;i<num_levels;i++)
    {
        imshow("Image level: "+to_string(i+1),imagePyramid[i]);
        waitKey(0);
        destroyWindow("Image level: "+to_string(i+1));
    }
}

//Image Pyramid
//
Size DeepPyramid::calculateLevelPyramidImageSize(int i)
{
    Size levelPyramidImageSize;
    if(originalImg.rows<=originalImg.cols)
    {
        levelPyramidImageSize.width=1713.0/pow(2,(num_levels-1-i)/2.0);
        levelPyramidImageSize.height=levelPyramidImageSize.width*(originalImg.rows/((double)originalImg.cols));
    }
    else
    {
        levelPyramidImageSize.height=1713.0/pow(2,(num_levels-1-i)/2.0);
        levelPyramidImageSize.width=levelPyramidImageSize.height*(originalImg.cols/((double)originalImg.rows));
    }

    return levelPyramidImageSize;
}

Mat DeepPyramid::createLevelPyramidImage(int i)
{
    Mat levelImg(1713,1713,CV_8UC3,Scalar(0,0,0));
    Size pictureSize=calculateLevelPyramidImageSize(i);
    Mat resizedImg;
    resize(originalImg, resizedImg, pictureSize);
    resizedImg.copyTo(levelImg(Rect(Point(0,0),pictureSize)));

    return levelImg;
}

void DeepPyramid::createImagePyramid()
{
    imagePyramid.clear();
    for(int i=0;i<num_levels;i++)
    {
        imagePyramid.push_back(createLevelPyramidImage(i));
    }
    cout<<"Create image pyramid..."<<endl;
    cout<<"Status: Success!"<<endl;
}
//
////

//NeuralNet
//
void DeepPyramid::calculateNet()
{
    net_->ForwardPrefilled();
}

cv::Mat DeepPyramid::convertToFloat(Mat img)
{
    cv::Mat img_float;
    if (img.channels() == 3)
    {
        img.convertTo(img_float, CV_32FC3);
    }
    else
    {
        if(img.channels()==1)
        {
            img.convertTo(img_float, CV_32FC1);
        }
    }

    return img_float;
}

void DeepPyramid::fillNeuralNetInput(int i)
{
    Mat img=imagePyramid[i];
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    net_->Reshape();

    std::vector<cv::Mat> input_channels;
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    cv::Mat img_float=convertToFloat(img);
    cv::split(img_float, input_channels);
}

void DeepPyramid::calculateNetAtLevel(int i)
{
    const clock_t begin_time = clock();
    cout<<"Calculate level: "+to_string(i+1)+" ..."<<endl;
    fillNeuralNetInput(i);
    calculateNet();
    cout << "Time:"<<float( clock () - begin_time ) /  CLOCKS_PER_SEC<<" s."<<endl;
    cout<<"Status: Success!"<<endl;

}

std::vector<Mat> DeepPyramid::wrapNetOutputLayer()
{
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    float* data=new float[output_layer->height()*output_layer->width()];

    std::vector<cv::Mat> max5Level;

    for(int k=0;k<output_layer->channels();k++)
    {
        for(int i=0;i<output_layer->height()*output_layer->width();i++)
        {
            data[i]=begin[i+output_layer->height()*output_layer->width()*k];
        }
        Mat conv(output_layer->height(),output_layer->width(), CV_32FC1, data);
        max5Level.push_back(conv.clone());
    }

    return max5Level;
}
//
////

//Max5
//
std::vector<Mat> DeepPyramid::createLevelPyramidMax5(int i)
{
    calculateNetAtLevel(i);
    return wrapNetOutputLayer();
}
void DeepPyramid::createMax5Pyramid()
{
    for(int i=0;i<num_levels;i++)
    {
        max5.push_back(createLevelPyramidMax5(i));
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
    meanStdDev(unite,mean,deviation);
    meanValue=mean.at<double>(0,0);
    deviationValue=deviation.at<double>(0,0);
}

std::vector<Mat> DeepPyramid::createLevelPyramidNorm5(int i)
{
    float mean,deviation;
    calculateMeanAndDeviationValue(max5[i],mean,deviation);
    vector<Mat> norm5;
    for(unsigned int j=0;j<max5[i].size();j++)
    {
        norm5.push_back((max5[i][j]-mean)/deviation);
    }

    return norm5;
}

void DeepPyramid::createNorm5Pyramid()
{
    for(unsigned int i=0;i<max5.size();i++)
    {
        norm5.push_back(createLevelPyramidNorm5(i));
    }
}
//
////

//Root-Filter sliding window
//
cv::Mat DeepPyramid::getFeatureVector(int levelIndx, Point position, Size size)
{
    cv::Mat feature(1,size.height*size.width*norm5[levelIndx].size(),CV_32FC1);
    for(int w=0;w<size.width;w++)
    {
        for(int h=0;h<size.height;h++)
        {
            for(unsigned int k=0;k<norm5[levelIndx].size();k++)
            {
                feature.at<float>(0,w+h*size.width+k*size.height*size.width)=norm5[levelIndx][k].at<float>(position.y+h,position.x+w);
            }
        }
    }
    return feature;

}

void DeepPyramid::rootFilterAtLevel(int rootFilterIndx, int levelIndx, int stride)
{
    Size filterSize=rootFilterSize[rootFilterIndx];
    CvSVM* filterSVM=rootFilterSVM[rootFilterIndx];
    int stepWidth, stepHeight;
    stepWidth=((norm5[levelIndx][0].cols-filterSize.width)/stride)+1;
    stepHeight=((norm5[levelIndx][0].rows-filterSize.height)/stride)+1;

    for(int w=0;w<stepWidth;w++)
        for(int h=0;h<stepHeight;h++)
        {
            Point p(stride*stepWidth,stride*stepHeight);
            cv::Mat feature=getFeatureVector(levelIndx,p,filterSize);
            int predict;
            double confidence;
            predict=filterSVM->predict(feature);
            confidence=filterSVM->predict(feature,true);
            FaceBox face;
            face.confidence=confidence;
            face.level=levelIndx;
            face.norm5Box=Rect(p,filterSize);
            if(predict==1)
            {
                face.type=FACE;
                allFaces.push_back(face);
                cout<<"lol"<<endl;
            }
            else
            {
                if(predict==-1)
                {
                    face.type=NOT_FACE;
                }
                //cout<<"here!"<<endl;
            }
        }
}

void DeepPyramid::rootFilterConvolution()
{
    for(unsigned int i=0;i<rootFilterSize.size();i++)
        for(unsigned int j=0;j<norm5.size();j++)
            rootFilterAtLevel(i,j,1);
}
//
////

//Rectangle
//
void DeepPyramid::calculateImagePyramidRectangle()
{
    for(unsigned int i=0;i<allFaces.size();i++)
    {
        allFaces[i].pyramidImageBox=getRectByNorm5Rect(allFaces[i].norm5Box);
    }
}

void DeepPyramid::calculateOriginalRectangle()
{
    calculateImagePyramidRectangle();
    float* scales=new float[num_levels];
    double longSide=(originalImg.cols>originalImg.rows) ? originalImg.cols : originalImg.rows;
    for(int i=0;i<num_levels;i++)
    {
        scales[i]=longSide/(1713/pow(2, (6-i)/2.0));
    }
    for(unsigned int i=0;i<allFaces.size();i++)
    {
        Rect original;
        Rect pyramid=allFaces[i].pyramidImageBox;
        int level=allFaces[i].level;
        original.x=pyramid.x*scales[level];
        original.y=pyramid.y*scales[level];
        original.width=pyramid.width*scales[level];
        original.height=pyramid.height*scales[level];
        allFaces[i].originalImageBox=original;
    }
}

void DeepPyramid::groupOriginalRectangle()
{
    for(unsigned int i=0;i<allFaces.size();i++)
    {
        detectedFaces.push_back(allFaces[i].originalImageBox);
    }
    cv::groupRectangles(detectedFaces,1,0.2);
}

void DeepPyramid::drawFace()
{
    originalImg.copyTo(originalImgWithFace);
    for(unsigned int i=0;i<detectedFaces.size();i++)
    {
        rectangle(originalImgWithFace,detectedFaces[i],Scalar(0,0,255));
    }
}

void DeepPyramid::detect(Mat img)
{
    setImg(img);
    createImagePyramid();
    createMax5Pyramid();
    createNorm5Pyramid();
    cout<<"filter"<<endl;
    rootFilterConvolution();
    cout<<"rectangle"<<endl;
    calculateOriginalRectangle();
    cout<<allFaces.size()<<endl;
    cout<<"group"<<endl;
    groupOriginalRectangle();
    cout<<"draw"<<endl;
    drawFace();
    imshow("RESULT", originalImgWithFace);
    waitKey(0);
}

void DeepPyramid::addRootFilter(Size filterSize, CvSVM* classifier)
{
    rootFilterSize.push_back(filterSize);
    rootFilterSVM.push_back(classifier);
}
