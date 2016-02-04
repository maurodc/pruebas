#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sys/time.h>
#include <stdio.h>

using namespace cv;

#ifdef TIME
  #include "utils/Time.hpp"
#endif


int main(int argc, char **argv){



    Mat img = imread("../src/004530.png");
    //Mat img = imread("../mon.jpg");
    std::vector<KeyPoint> kp;

    Ptr<FeatureDetector> detector = FeatureDetector::create("ORB"); //detector generico
    detector->set("nFeatures", 105);
 
    //Cronometrando las operaciones
    #ifdef TIME
    double startStep, endStep;
    startStep = GetSeg();
    #endif

    detector->detect(img, kp);

    #ifdef TIME
    endStep = GetSeg();
    std::cout << "\n>>> Tiempo Consumido: " << endStep - startStep << std::endl;
    #endif
    std::cout << ">>> Se detectaron " << kp.size() << " Keypoints" << std::endl;
    
    Mat out;  
    drawKeypoints(img, kp, out, Scalar::all(-1)); //dibujo los features en la img
    imshow("Intento 1 - KeyPoints", out);  //Muestro la img con los features

    
    Mat descriptors;
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("ORB");
    extractor->compute(img, kp, descriptors);
   // std::cout << ">>> Se detectaron los siguientes descriptores \n" << descriptors << std::endl;
    
    waitKey(0);
    return 0;
}


 
