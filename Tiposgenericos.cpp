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

/*
Se creó un FeatureDetector de tipo: ORB
>>> Parámetro: WTA_K de tipo: int y  con valor: 2
>>> Parámetro: edgeThreshold de tipo: int y  con valor: 31
>>> Parámetro: firstLevel de tipo: int y  con valor: 0
>>> Parámetro: nFeatures de tipo: int y  con valor: 500
>>> Parámetro: nLevels de tipo: int y  con valor: 8
>>> Parámetro: patchSize de tipo: int y  con valor: 31
>>> Parámetro: scaleFactor de tipo: double y  con valor: 1.2
>>> Parámetro: scoreType de tipo: int y  con valor: 0
*/


// Se creó un FeatureDetector de tipo: FAST
// >>> Parámetro: nonmaxSuppression de tipo: bool y  con valor: 1
// >>> Parámetro: threshold de tipo: int y  con valor: 10


// Se creó un FeatureDetector de tipo: GFTT
// >>> Parámetro: k de tipo: double y  con valor: 0.04
// >>> Parámetro: minDistance de tipo: double y  con valor: 1
// >>> Parámetro: nfeatures de tipo: int y  con valor: 1000
// >>> Parámetro: qualityLevel de tipo: double y  con valor: 0.01
// >>> Parámetro: useHarrisDetector de tipo: bool y  con valor: 0




void setearParam(cv::Ptr<cv::FeatureDetector>& detector, YAML::Node nodo){

    
  if (nodo["Name"].as<string>()=="ORB")
  {
    detector->set("WTA_K", nodo["WTA_K"].as<int>());
    detector->set("edgeThreshold", nodo["edgeThreshold"].as<int>());
    detector->set("firstLevel", nodo["firstLevel"].as<int>());
    detector->set("nFeatures", nodo["nFeatures"].as<int>());
    detector->set("nLevels", nodo["nLevels"].as<int>());
    detector->set("patchSize", nodo["patchSize"].as<int>());
    detector->set("scaleFactor", nodo["scaleFactor"].as<double>());
    detector->set("scoreType", nodo["scoreType"].as<int>());
  }
  else if (nodo["Name"].as<string>()=="GFTT")
  {
    detector->set("k", nodo["k"].as<double>());
    detector->set("minDistance", nodo["minDistance"].as<double>());
    detector->set("nfeatures", nodo["nfeatures"].as<int>());
    detector->set("qualityLevel", nodo["qualityLevel"].as<double>());
    detector->set("useHarrisDetector", nodo["useHarrisDetector"].as<bool>());
  }
  else if (nodo["Name"].as<string>()=="FAST")
  {
    detector->set("nonmaxSuppression", nodo["nonmaxSuppression"].as<bool>());
    detector->set("threshold", nodo["threshold"].as<double>());
  }
  else
  std::cerr << "Parámetro no soportado - No existe FeatureDetector con el nombre "<< nodo["Name"].as<string>() << std::endl;
}


string identParamType(int type) {
        switch (type) {
        case cv::Param::BOOLEAN:
            return "bool";
            break;
        case cv::Param::INT:
            return "int";
            break;
        case cv::Param::REAL:
            return "double";
            break;
        case cv::Param::STRING:
            return "string";
            break;
        case cv::Param::MAT:
            return "Mat";
            break;
        case cv::Param::ALGORITHM:
            return "Algorithm";
            break;
        case cv::Param::MAT_VECTOR:
            return "Mat vector";
            break;
        }
}


void printParameters(cv::Ptr<cv::FeatureDetector> algorithm)
{
  std::vector<cv::String> parameters;
  algorithm->getParams(parameters); 

  for (int i=0; i<parameters.size();i++){
        std::cout << ">>> Parámetro: "<< parameters[i] << " de tipo: "<< identParamType(algorithm->paramType(parameters[i])) << " y  con valor: " << algorithm->getDouble(parameters[i]) << std::endl;    
   }

}


void loadConfiguration(std::string path, cv::Ptr<cv::FeatureDetector>& detector){

  //"../configurationFiles/kitti.yaml"
  YAML::Node config = YAML::LoadFile(path);
  
  // std::cout << ">>> Cargando archivo: " << path << std::endl;

  if(config["FeatureDetector"]){
    string name = config["FeatureDetector"]["Name"].as<string>();
    detector = cv::FeatureDetector::create( name );
    std::cout << "Se creó un FeatureDetector de tipo: " << name << std::endl;
    setearParam(detector, config["FeatureDetector"]);
  }
  else
    std::cout << "ERROR - No existe configuración para FeatureDetector" << std::endl;


}





int main(int argc, char **argv){

  cv::Ptr<cv::FeatureDetector> detector;
    
  loadConfiguration(argv[1], detector);    

  Mat img = imread("../src/004530.png");

  std::vector<KeyPoint> kp;

  

  //Imprimo la lista de parámetros para mi tipo de detector
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