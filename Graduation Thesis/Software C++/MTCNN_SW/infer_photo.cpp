#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

#include "detector.h"
#include "draw.hpp"

int main(int argc, char** argv) {
    std::string modelPath = "C:/Minh_Duc/MD_Personal/LVTN/mtcnn-opencv-master/models";

    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = "C:/Minh_Duc/MD_Personal/LVTN/mtcnn-opencv-master/models/det1.caffemodel";
    pConfig.protoText = "C:/Minh_Duc/MD_Personal/LVTN/mtcnn-opencv-master/models/det1.prototxt";
    pConfig.threshold = 0.6f;

    RefineNetwork::Config rConfig;
    rConfig.caffeModel = "C:/Minh_Duc/MD_Personal/LVTN/mtcnn-opencv-master/models/det2.caffemodel";
    rConfig.protoText = "C:/Minh_Duc/MD_Personal/LVTN/mtcnn-opencv-master/models/det2.prototxt";
    rConfig.threshold = 0.7f;

    OutputNetwork::Config oConfig;
    oConfig.caffeModel = "C:/Minh_Duc/MD_Personal/LVTN/mtcnn-opencv-master/models/det3.caffemodel";
    oConfig.protoText = "C:/Minh_Duc/MD_Personal/LVTN/mtcnn-opencv-master/models/det3.prototxt";
    oConfig.threshold = 0.7f;

    MTCNNDetector detector(pConfig, rConfig, oConfig);
    cv::Mat img = cv::imread("C:\\Minh_Duc\\Metro_dataset\\5555.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(1280, 720));
    cv::imshow("ddd", img);

    std::vector<Face> faces;
    faces = detector.detect(img, 20.f, 0.709f);
    std::cout << "Number of faces found in the supplied image - " << faces.size() << std::endl;

    std::vector<rectPoints> data;


    std::string outputFolder = "C:\\Minh_Duc\\MD_Personal\\LVTN\\MTCNN_SW\\Dataset\\output\\MTCNN_IMG";

    for (size_t i = 0; i < faces.size(); ++i) {
        std::vector<cv::Point> pts;
        for (int p = 0; p < NUM_PTS; ++p) {
            pts.push_back(cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
        }

        auto rect = faces[i].bbox.getRect();
        auto d = std::make_pair(rect, pts);
        data.push_back(d);

        cv::Rect faceRect(faces[i].bbox.getRect());
        cv::Mat faceImg = img(faceRect);

        cv::Mat resizedFace;
        cv::resize(faceImg, resizedFace, cv::Size(200, 200));

        std::string windowName = "Face " + std::to_string(i);
        cv::imshow(windowName, resizedFace);

        std::string faceImagePath = outputFolder + "/face_" + std::to_string(i) + ".jpg";
        if (!cv::imwrite(faceImagePath, resizedFace)) {
            std::cerr << "Failed to save image: " << faceImagePath << std::endl;
            return -1;
        }
        else {
            std::cout << "Saved image: " << faceImagePath << std::endl;
        }
    }

    auto resultImg = drawRectsAndPoints(img, data);
    cv::imshow("Detected Faces", resultImg);
    cv::waitKey(0);

    return 0;
}