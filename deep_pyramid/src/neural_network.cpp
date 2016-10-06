// Copyright 2016 Dolotov Evgeniy

#include "../include/neural_network.h"
#include <string>
#include <vector>
using std::string;
using std::vector;
using cv::Mat;
using cv::Size;
using caffe::Net;
using caffe::Blob;
using caffe::Caffe;
NeuralNetwork::NeuralNetwork(string configFile, string trainedModel) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    net.reset(new Net<float>(configFile, caffe::TEST));
    net->CopyTrainedLayersFrom(trainedModel);
    Blob<float>* input_layer = net->input_blobs()[0];
    assert(input_layer->width() == input_layer->height());
}

void NeuralNetwork::processImage(const Mat &img, FeatureMap& map) {
    fillNeuralNetInput(img);
    calculate();
    getNeuralNetOutput(map);
}

void NeuralNetwork::fillNeuralNetInput(const Mat &img) {
    Blob<float>* input_layer = net->input_blobs()[0];
    int width = input_layer->width();
    int height = input_layer->height();
    input_layer->Reshape(1, input_layer->channels(), height, width);
    net->Reshape();

    vector<Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }

    Mat img_float;
    img.convertTo(img_float, CV_32FC3);
    split(img_float, input_channels);
}

void NeuralNetwork::getNeuralNetOutput(FeatureMap &map) {
    Blob<float>* output_layer = net->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    float* data = new float[output_layer->height()*output_layer->width()];

    for (int k = 0; k < output_layer->channels(); k++) {
        for (int i  = 0; i < output_layer->height()*output_layer->width();  i++) {
            data[i] = begin[i+output_layer->height()*output_layer->width()*k];
        }
        Mat conv(output_layer->height(), output_layer->width(), CV_32FC1, data);
        map.addLayer(conv.clone());
    }
}

void NeuralNetwork::calculate() {
    net->ForwardPrefilled();
}

Size NeuralNetwork::inputLayerSize()  {
    Blob<float>* input_layer = net->input_blobs()[0];
    return Size(input_layer->width(), input_layer->height());
}

Size NeuralNetwork::outputLayerSize() {
    Blob<float>* output_layer = net->output_blobs()[0];
    return Size(output_layer->width(), output_layer->height());
}
