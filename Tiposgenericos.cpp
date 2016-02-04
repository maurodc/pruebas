#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sys/time.h>
#include <stdio.h>


#include <yaml-cpp/yaml.h>


using namespace cv;

#ifdef TIME
  #include "utils/Time.hpp"
#endif


cv::Ptr<cv::FeatureDetector> createFeatureDetector( const YAML::Node& node )
{
  const std::string name = node["Name"].as<std::string>();
  std::cout << "detector: " << name << std::endl;

  cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create( name );

  if ( not detector ) {
    std::cout << "could not load detector with name " << name << std::endl;
    return nullptr;
  }

  setParameters( detector, node );

  return detector;

  ///Previamente: 
  ///ProgramOptions program_options( argv[0] );
  //program_options.addPositionalArgument("configuration", "configuration file with SPTAM parameters.", parametersFileYML);


   /// al invocar descriptorExtractor = loadDescriptorExtractor( config["DescriptorExtractor"] );

}


void printParameters(cv::Ptr<cv::FeatureDetector> algorithm)
{
  std::vector<cv::String> parameters;
  algorithm->getParams(parameters); 

  for (int i=0; i<parameters.size();i++){
        std::cout << ">>> Parámetro: "<< parameters[i] << std::endl;    
   }

}

int main(int argc, char **argv){

    //YAML::Node config = YAML::LoadFile("../configurationFiles/kitti.yaml");

    Mat img = imread("../src/004530.png");
    //Mat img = imread("../mon.jpg");
    std::vector<KeyPoint> kp;

    Ptr<FeatureDetector> detector = FeatureDetector::create("GFTT"); //detector generico
    //detector->set("nFeatures", 500);
    detector->set("nfeatures", 500);
    printParameters(detector);  

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

