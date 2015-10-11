#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <math.h>
#include <caffe/caffe.hpp>
#include <caffe/common.hpp>
#include <vector>

using namespace cv;
using namespace std;
using namespace caffe;

class DeepPyramid
{
public:
    int num_levels;
    std::vector< std::vector<Mat> > levels;
    shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    DeepPyramid(int num_levels, const string& model_file,
               const string& trained_file);
    std::vector<cv::Mat> calculate_level(const cv::Mat& img);

    void WrapInputLayer(std::vector<cv::Mat>* input_channels);

    void Preprocess(const cv::Mat& img,
                    std::vector<cv::Mat>* input_channels);
    void calculate(const cv::Mat& img);
};

 DeepPyramid:: DeepPyramid(int num_levels, const string& model_file,
                       const string& trained_file) {
    this->num_levels=num_levels;
  Caffe::set_mode(Caffe::CPU);
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

std::vector<cv::Mat> DeepPyramid::calculate_level(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();

  float* data=new float[output_layer->height()*output_layer->width()];
cout<<output_layer->channels()<<endl;
cout<<output_layer->height()<<endl;
cout<<output_layer->width()<<endl;
std::vector<cv::Mat> result;
for(int k=0;k<output_layer->channels();k++)
{
  for(int i=0;i<output_layer->height()*output_layer->width();i++)
  {
      data[i]=begin[i+output_layer->height()*output_layer->width()*k];
  }
  Mat conv(output_layer->height(),output_layer->width(), CV_32FC1, data);
  result.push_back(conv.clone());
}
  return result;
}


void DeepPyramid::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void DeepPyramid::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);
  cv::split(sample_float, *input_channels);
}

void DeepPyramid::calculate(const Mat &img)
{
    int imgHeight, imgWidth;
    imgHeight=img.rows;
    imgWidth=img.cols;
    Mat* imgPyramid=new Mat[num_levels];
    for(int i=0;i<num_levels;i++)
    {
        imgPyramid[i]=Mat(1713,1713,CV_8UC3,Scalar(0,0,0));
        int newImgHeight, newImgWidth;
        if(imgHeight<=imgWidth)
        {
            newImgWidth=1713.0/pow(2,(num_levels-1-i)/2.0);
            newImgHeight=newImgWidth*(imgHeight/((double)imgWidth));
        }
        else
        {
            newImgHeight=1713.0/pow(2,(num_levels-1-i)/2.0);
            newImgWidth=newImgHeight*(imgWidth/((double)imgHeight));
        }
        Mat resizedImg;
        resize(img, resizedImg,Size(newImgWidth,newImgHeight));
        Rect rect(0,0,newImgWidth,newImgHeight);
        resizedImg.copyTo(imgPyramid[i](rect));
    }

    for(int i=0;i<num_levels;i++)
    {
        levels.push_back(this->calculate_level(imgPyramid[i]));
    }
}
cv::Mat getFeatureVector(std::vector<cv::Mat> level, cv::Point position, cv::Size size)
{
    cv::Mat feature(1,size.height*size.width*level.size(),CV_32FC1);
    for(int k=0;k<level.size();k++)
    {
        for(int w=0;w<size.width;w++)
        {
            for(int h=0;h<size.height;h++)
                feature.at<float>(1,w+h*size.height+k*size.height*size.width)=level[k].at<float>(position.x+w,position.y+h);
        }
    }
    return feature;
}
cv::Mat DpmSlidingWindow(std::vector<cv::Mat> level, CvSVM filter, int stride, cv::Size size)
{
    int scoreWidth, scoreHeight;
    scoreWidth=(level[0].cols-size.width)/stride+1;
    scoreHeight=(level[0].cols-size.height)/stride+1;
    Mat score(scoreWidth,scoreHeight,CV_32FC1);
    for(int i=0;i<scoreWidth;i++)
    {
        for(int j=0;j<scoreHeight;j++)
        {
            Mat feature=getFeatureVector(level,Point(stride*i,stride*j),size);
            score.at<float>(i,j)=filter.predict(feature);
        }
    }
    return score;
}

int main(int argc, char *argv[])
{
    Caffe::set_mode(Caffe::CPU);
    Mat image;

    string alexnet_model_file=argv[1];
    string alexnet_trained_file=argv[2];

    DeepPyramid pyramid(7,alexnet_model_file, alexnet_trained_file);

    string image_file=argv[3];

    image=imread(image_file, CV_LOAD_IMAGE_COLOR);
    namedWindow("Image", WINDOW_AUTOSIZE);
    imshow("Image",image);
    waitKey(0);
    pyramid.calculate(image);
    return 0;
}
