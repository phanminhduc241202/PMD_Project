#pragma once
#ifndef _include_opencv_mtcnn_detector_h_
#define _include_opencv_mtcnn_detector_h_

#include "face.h"
#include "O_NET.h"
#include "P_NET.h"
#include "R_NET.h"

class MTCNNDetector {
private:
    std::unique_ptr<ProposalNetwork> _pnet;
    std::unique_ptr<RefineNetwork> _rnet;
    std::unique_ptr<OutputNetwork> _onet;

public:
    MTCNNDetector(const ProposalNetwork::Config& pConfig,
        const RefineNetwork::Config& rConfig,
        const OutputNetwork::Config& oConfig);
    std::vector<Face> detect(const cv::Mat& img, const float minFaceSize,
        const float scaleFactor);
};

#endif