#include <iostream>
#include <list>
#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

#include "variance.h"
int main(int argc, char ** argv){
  Mat img = imread(argv[1]);
  if(img.empty()){
    cout << "Error : Couldn't open " << argv[1]  << endl;
    return 1;
  }
  resize(img, img, Size(640, 480));
  imwrite("var.png", img);  
  variance var;
  list<Rect> rects;
  var.detect(img, rects);
  var.draw(img);
  imwrite("var.png", img);
  return 0;
}
