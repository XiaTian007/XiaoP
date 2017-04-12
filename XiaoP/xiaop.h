#ifndef XIAOP_H
#define XIAOP_H
#include <opencv2/core/core.hpp>  
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <QtWidgets/QMainWindow>
#include "ui_xiaop.h"
#include <QFileDialog>  
#include <QMessageBox>
using namespace cv;
class XiaoP : public QMainWindow
{
	Q_OBJECT

public:
	//理想低通
	static const int IDEAL_LOW = 1;
	//理想高通
	static const int IDEAL_HIGH = 2;
	//巴特沃斯低通
	static const int BW_LOW = 3;
	//巴特沃斯高通
	static const int BW_HIGH = 4;
	XiaoP(QWidget *parent = 0);
	~XiaoP();
	//归一化，将灰度映射到0~255之间, 并将能量最高的四角移到中心, 生成图片频域能量图
	void BuildDFTImage(IplImage *fourier, IplImage *dst);
	//DFT变换
	IplImage *DFT(IplImage * src);
	//几何均值滤波器——模板大小5*5
	IplImage* GeometryMeanFilter(IplImage* src);
	//RGBToGray
	IplImage* XiaoP::Gray(IplImage* src);
	//谐波均值滤波器——模板大小5*5
	IplImage* HarmonicMeanFilter(IplImage* src);
	//DFT反变换
	IplImage *IDFT(IplImage * fourier);
	//逆谐波均值大小滤波器——模板大小5*5
	IplImage* InverseHarmonicMeanFilter(IplImage* src);
	//中值滤波 模板大小5*5
	IplImage* MedianFilter_5_5(IplImage* src);
	//中值滤波 模板大小9*9
	IplImage* MedianFilter_9_9(IplImage* src);

	/** 频域滤波函数
	* @parameter
	* fourier 傅里叶变换图像，二通道
	* FlAG 全局变量编辑，确定用哪个滤波器
	* d0 滤波器半径
	* n1 巴特沃斯滤波器参数
	*/
	void PassFilter(IplImage * fourier, int FLAG, double d0, int n1);

	//Roberts算子实现
	Mat roberts(Mat srcImage);
	//自适应均值滤波
	IplImage* SelfAdaptMeanFilter(IplImage* src);
	//自适应中值滤波
	IplImage* SelfAdaptMedianFilter(IplImage* src);
	//手动实现拉普拉斯算子图像锐化  
	void sharpenImage1(const Mat &image, Mat &result);
	//调用OpenCV函数实现拉普拉斯算子图像锐化  
	void sharpenImage2(const Mat &image, Mat &result);
	//5*5模板通用函数
	Mat Templet(Mat img, const int(*templet)[5], int normalize);

	//获得直方图分布图
	Mat getHistImg(Mat img);
	//直方图均衡化
	Mat Equalization(Mat img);

	IplImage* AddGuassianNoise(IplImage* src);    //添加高斯噪声
	IplImage* AddPepperNoise(IplImage* src);      //添加胡椒噪声，随机黑色点
	IplImage* AddPepperSaltNoise(IplImage* src);    //添加椒盐噪声，随机黑白点
	IplImage* AddSaltNoise(IplImage* src);       //添加盐噪声，随机白色点
	IplImage* ArithmeticMeanFilter(IplImage* src);//算术均值滤波器——模板大小5*5
	void img_display(cv::Mat mat);//右边显示图像函数，Mat格式
	void img_display(IplImage* mat);//右边显示图像函数，IplImage格式
	void dst_display(cv::Mat mat);//左边显示图像函数，Mat格式
	void dst_display(IplImage* mat);//左边显示图像函数，IplImage格式
	
private slots:
	void on_openButton_clicked();//打开图像按钮程序
	void on_grayButton_clicked();//彩色图像转为灰度图
	void on_histButton_clicked();//显示直方图
	void on_histEqualButton_clicked();//直方图均衡
	void on_oriButton_clicked();//显示原图
	void on_equalButton_clicked();//均值滤波
	void on_medianButton_clicked();//中值滤波
	void on_gaussButton_clicked();//高斯滤波
	void on_saltButton_clicked();//加盐噪声
	void on_pepperButton_clicked();//胡椒噪声
	void on_pepperSaltButton_clicked();//椒盐噪声
	void on_gaussNoiseButton_clicked();//高斯噪声
	void on_sharpButton_clicked();//锐化图像
	void on_DFTButton_clicked();//DFT变换
	void on_LPButton_clicked();//低通滤波
	void on_laplaceButton_clicked();//拉普拉斯图

private:

	Ui::XiaoPClass ui;
};

#endif // XIAOP_H
