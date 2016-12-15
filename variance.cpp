#include <iostream>
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#include "variance.h"

variance::variance():
m_wsz(16), m_th(30),
m_roi_rc(0, 0, -1, -1), m_cmp(true)
{}


inline void variance::calc_vrnc_map(const cv::Mat &src, const cv::Mat &sum,
			  const cv::Mat &sqsum, const cv::Mat  &mask, cv::Mat &dst)
{
  cout <<"Entering calc_vrnc_map" << endl;
  dst = cv::Mat::zeros(src.rows, src.cols, CV_64F);
  const double  isqwsz = 1/(double)(m_wsz*m_wsz);
  const int xmax = src.cols - m_wsz;
 // const int ymax = src.rows - m_wsz + 1 - src.rows/2;
  const int ymax = src.rows - m_wsz;
  for(int y = 0; y < ymax; ++y){
    double *pdst = dst.ptr<double>(y);
    const int ywsz = y + m_wsz;
    const int *psum0 = sum.ptr<int>(y);
    const int *psum1 = sum.ptr<int>(ywsz);
    const double *psqsum0 = sqsum.ptr<double>(y);
    const double *psqsum1 = sqsum.ptr<double>(ywsz);
	const uchar *pmask = mask.ptr<uchar>(ywsz);
    for(int x = 0; x < xmax; ++x){
      const int xwsz = x+m_wsz;
      const double ave_sq = (psum1[xwsz] - psum1[x] - psum0[xwsz] + psum0[x])*isqwsz; 
      pdst[x] = (psqsum1[xwsz] - psqsum1[x] - psqsum0[xwsz] + psqsum0[x])*isqwsz - ave_sq*ave_sq; 
	  pdst[x] *= pmask[x];
	}
  }
  cout << "Exiting vrnc_map " << endl;
}

inline void variance::addRect(cv::Rect rc)
{
  if(m_cmp_rcs.size() == 0){
    m_cmp_rcs.push_back(rc);
    return;
  }

  list<cv::Rect>::iterator it = m_cmp_rcs.begin();
  for(;!(it == m_cmp_rcs.end());){
    cv::Rect new_rc;
    new_rc.x = rc.x < it->x ? rc.x : it->x;
    
    const int rc0_right = rc.x + rc.width;
    const int rc1_right = it->x + it->width;
    new_rc.width = rc0_right > rc1_right ? rc0_right - new_rc.x: rc1_right - new_rc.x;

    if(new_rc.width > rc.width + it->width){
      ++it;
      continue;
    }
    new_rc.y = rc.y < it->y ? rc.y : it->y;
    
    const int rc0_upper = rc.y + rc.height;
    const int rc1_upper = it->y + it->height;
    new_rc.height = rc0_upper > rc1_upper ? rc0_upper - new_rc.y: rc1_upper - new_rc.y;

    if(new_rc.height > rc.height + it->height){
      ++it;
      continue;
    }

    m_cmp_rcs.erase(it);
    rc = new_rc;
    it = m_cmp_rcs.begin();
  }
  m_cmp_rcs.push_back(rc);
}

inline void variance::detectRect(const cv::Mat &dst, const double th, 
				   const double wsz)
{
  const double *pdst = dst.ptr<double>(0);
  for(int y = 0; y < dst.rows; ++y){
    for(int x = 0; x < dst.cols; ++x){
      if(pdst[x] > th){
	m_cmp_rcs.push_back(cv::Rect(x, y, (int)wsz, (int)wsz));
      }
    }
  }
}

void variance::detect(const Mat &src_img, list<Rect> &rects){
  Mat hsv_img;
  cvtColor(src_img, hsv_img, CV_BGR2HSV);
  vector<Mat> planes;
  split(hsv_img, planes);
  Mat img;
  img=planes[0];
  Mat mask = Mat(img.size(), CV_8U, Scalar(1));
  cv::Mat sum, sqsum, var;
  cv::integral(img, sum, sqsum, CV_32S);
  calc_vrnc_map(img, sum, sqsum,  mask, var);
  if(m_cmp){
    for(int y = 0; y < var.rows/3; ++y){
      const double *pvar = var.ptr<double>(y);
      for(int x = 0; x < var.cols; ++x){
	if(pvar[x] > m_th)
	  addRect(cv::Rect(x, y, m_wsz, m_wsz));
      }
    }
  }
  else{
    for(int y = 0; y < var.rows/2; ++y){
      const double *pvar = var.ptr<double>(y);
      for(int x = 0; x < var.cols; ++x){
	if(pvar[x] > m_th)
	  m_cmp_rcs.push_back(cv::Rect(x, y, m_wsz, m_wsz));
      }
    }
  }

  rects = m_cmp_rcs;
}

void variance::draw(Mat &dst){
  cout << m_cmp_rcs.size() << endl;
  for (list<cv::Rect>::iterator it = m_cmp_rcs.begin();
       !(it == m_cmp_rcs.end()); ++it) {
    rectangle(dst, *it, cv::Scalar(0, 255, 0));
  }
}
