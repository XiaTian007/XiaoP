#include "xiaop.h"
#include <vector>
using namespace cv;

//均值5*5模板
static const int mean2[5][5] = { { 1,1,1,1,1 },{ 1,1,1,1,1 },{ 1,1,1,1,1 },{ 1,1,1,1,1 },{ 1,1,1,1,1 } };
//高斯5*5模板
static const int Gauss2[5][5] = { { 1,4,7,4,1 } ,{ 4,16,26,16,4 } ,
{ 7,26,41,26,7 },{ 4,16,26,16,4 },{ 1,4,7,4,1 } };

//全局图像变量，用于恢复原图,只在打开图像中赋值
static  IplImage* ori_img;
//全局图像变量，左边图像
static  IplImage* left_img;
//全局图像变量，右边图像
static  IplImage* right_img;

XiaoP::XiaoP(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

XiaoP::~XiaoP()
{

}


//打开图像按钮程序
void XiaoP::on_openButton_clicked() {
	QString fileName = QFileDialog::getOpenFileName(
		this, "open image file",
		".",
		"Image files (*.bmp *.jpg *.pbm *.pgm *.png *.ppm *.xbm *.xpm);;All files (*.*)");

	ori_img = cvLoadImage(fileName.toUtf8().constData());
	left_img = cvCreateImage(cvGetSize(ori_img), IPL_DEPTH_8U, ori_img->nChannels);
	right_img = cvCreateImage(cvGetSize(ori_img), IPL_DEPTH_8U, ori_img->nChannels);
	cvCopy(ori_img, left_img);
	cvCopy(ori_img, right_img);
	img_display(left_img);
}

//彩色图像转为灰度图
void XiaoP::on_grayButton_clicked() {
	if (ori_img->nChannels == 3) {
		left_img = Gray(ori_img);
		img_display(left_img);
	}
	
}
//显示直方图
void XiaoP::on_histButton_clicked() {
	Mat temp = cvarrToMat(left_img, true);
	Mat hist_left;
	hist_left.create(temp.size(), CV_8UC1);
	hist_left = getHistImg(temp);
	namedWindow("左图直方图");
	imshow("左图直方图", hist_left);
	temp = cvarrToMat(right_img, true);
	Mat hist_right;
	hist_right.create(temp.size(), CV_8UC1);
	hist_right = getHistImg(temp);
	namedWindow("右图直方图");
	imshow("右图直方图", hist_right);
}
//直方图均衡
void XiaoP::on_histEqualButton_clicked() {
	Mat temp = cvarrToMat(left_img);
	Mat hist_e = Equalization(temp);
	dst_display(hist_e);
}
//显示原图
void XiaoP::on_oriButton_clicked() {
	left_img = cvCreateImage(cvGetSize(ori_img), IPL_DEPTH_8U, ori_img->nChannels);
	cvCopy(ori_img, left_img);
	img_display(left_img);
}
//均值滤波
void XiaoP::on_equalButton_clicked() {
	IplImage* img_m = ArithmeticMeanFilter(left_img);
	dst_display(img_m);
}
//中值滤波
void XiaoP::on_medianButton_clicked() {
	IplImage* img_m = MedianFilter_5_5(left_img);
	dst_display(img_m);
}
//高斯滤波
void XiaoP::on_gaussButton_clicked() {
	Mat temp = cvarrToMat(left_img, true);
	Mat img_g;
	if (left_img->nChannels == 3) {
		Mat pImageChannel[3];
		int i;
		for (i = 0; i < 3; i++) {
			pImageChannel[i].create(cvGetSize(left_img), CV_8UC1);
		}
		std::vector<Mat> src;
		split(temp, pImageChannel);
		for (i = 0; i < 3; i++) {
			pImageChannel[i] = Templet(pImageChannel[i], Gauss2, 273);
			src.push_back(pImageChannel[i]);
		}
		
		img_g.create(cvGetSize(left_img), CV_8UC3);
		merge(src,img_g);
	}
	else {
		img_g = Templet(temp, Gauss2, 273);
	}
	dst_display(img_g);
}
//加盐噪声
void XiaoP::on_saltButton_clicked() {
	left_img = AddSaltNoise(left_img);
	img_display(left_img);
}
//胡椒噪声
void XiaoP::on_pepperButton_clicked() {
	left_img = AddPepperNoise(left_img);
	img_display(left_img);
}
//椒盐噪声
void XiaoP::on_pepperSaltButton_clicked() {
	left_img = AddPepperSaltNoise(left_img);
	img_display(left_img);
}
//高斯噪声
void XiaoP::on_gaussNoiseButton_clicked() {
	left_img = AddGuassianNoise(left_img);
	img_display(left_img);
}
//锐化图像
void XiaoP::on_sharpButton_clicked() {
	Mat temp = cvarrToMat(left_img, true);
	Mat temp1 = cvarrToMat(right_img, true);
	sharpenImage2(temp, temp1);
	dst_display(temp1);
}
//DFT变换
void XiaoP::on_DFTButton_clicked() {
	IplImage* fourier = DFT(left_img);
	IplImage* f_c = cvCreateImage(cvGetSize(fourier), IPL_DEPTH_8U, 1);
	BuildDFTImage(fourier, f_c);
	dst_display(f_c);
	cvNamedWindow("DFT");
	cvShowImage("DFT", f_c);
}
//低通滤波
void XiaoP::on_LPButton_clicked() {
	if (left_img->nChannels == 1) {
		IplImage* fourier = cvCreateImage(cvGetSize(left_img), IPL_DEPTH_8U, 2);
		fourier = DFT(left_img);
		PassFilter(fourier, IDEAL_LOW, 100, 2);
		IplImage* dst_img = cvCreateImage(cvGetSize(left_img), IPL_DEPTH_8U, 1);
		dst_img = IDFT(fourier);
		left_img = cvCreateImage(cvGetSize(left_img), IPL_DEPTH_8U, 1);
		cvCopy(dst_img, left_img);
		img_display(left_img);
	}
	else {
		QMessageBox::warning(NULL, "warning", "Please turn to gray first", QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes);
	}
}
//拉普拉斯图
void XiaoP::on_laplaceButton_clicked() {
	Mat dst_img,outImg;
	Mat temp = cvarrToMat(ori_img, true);
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	if (temp.channels() == 3) {
		dst_img = cvarrToMat(Gray(ori_img),true);
	}
	else {
		dst_img = temp;
	}
	//拉普拉斯算子
	Laplacian(dst_img, outImg, ddepth, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(outImg, outImg);
	dst_display(outImg);
}

//RGBToGray
IplImage* XiaoP::Gray(IplImage* src) {

	//注意指针变量一定要先初始化才能使用，否则崩溃  
	//灰度转换时通道一定要设置正确  
	int channel = 1;//image->nChannels;  
	int depth = src->depth;
	CvSize sz;
	sz.width = src->width;//宽度  
	sz.height = src->height;//高度  
	IplImage* result;
	result = cvCreateImage(sz, depth, channel);//生成image 
	cvCvtColor(src, result, CV_BGR2GRAY);
	return result;
}

//Roberts算子实现
Mat XiaoP::roberts(Mat srcImage)
{
	Mat dstImage = srcImage.clone();
	int nRows = dstImage.rows;
	int nCols = dstImage.cols;
	for (int i = 0; i < nRows - 1; i++)
	{
		for (int j = 0; j < nCols - 1; j++)
		{
			int t1 = (srcImage.at<uchar>(i, j) -
				srcImage.at<uchar>(i + 1, j + 1)) *
				(srcImage.at<uchar>(i, j) -
					srcImage.at<uchar>(i + 1, j + 1));
			int t2 = (srcImage.at<uchar>(i + 1, j) -
				srcImage.at<uchar>(i, j + 1)) *
				(srcImage.at<uchar>(i + 1, j) -
					srcImage.at<uchar>(i, j + 1));
			dstImage.at<uchar>(i, j) = (uchar)sqrt(t1 + t2);

		}
	}
	return dstImage;
}

//手动实现拉普拉斯算子图像锐化  
void XiaoP::sharpenImage1(const Mat &image, Mat &result)
{
	result.create(image.size(), image.type());//为输出图像分配内容  
											  /*拉普拉斯滤波核3*3
											  0  -1   0
											  -1   5  -1
											  0  -1   0  */
											  //处理除最外围一圈外的所有像素值  
	for (int i = 1; i<image.rows - 1; i++)
	{
		const uchar * pre = image.ptr<const uchar>(i - 1);//前一行  
		const uchar * cur = image.ptr<const uchar>(i);//当前行，第i行  
		const uchar * next = image.ptr<const uchar>(i + 1);//下一行  
		uchar * output = result.ptr<uchar>(i);//输出图像的第i行  
		int ch = image.channels();//通道个数  
		int startCol = ch;//每一行的开始处理点  
		int endCol = (image.cols - 1)* ch;//每一行的处理结束点  
		for (int j = startCol; j < endCol; j++)
		{
			//输出图像的遍历指针与当前行的指针同步递增, 以每行的每一个像素点的每一个通道值为一个递增量, 因为要  

			//考虑到图像的通道数
			//saturate_cast<uchar>保证结果在uchar范围内  
			*output++ = saturate_cast<uchar>(5 * cur[j] - pre[j] - next[j] - cur[j - ch] - cur[j + ch]);
		}
	}
	//将最外围一圈的像素值设为0  
	result.row(0).setTo(Scalar(0));
	result.row(result.rows - 1).setTo(Scalar(0));
	result.col(0).setTo(Scalar(0));
	result.col(result.cols - 1).setTo(Scalar(0));
	/*/或者也可以尝试将最外围一圈设置为原图的像素值
	image.row(0).copyTo(result.row(0));
	image.row(image.rows-1).copyTo(result.row(result.rows-1));
	image.col(0).copyTo(result.col(0));
	image.col(image.cols-1).copyTo(result.col(result.cols-1));*/
}

//调用OpenCV函数实现拉普拉斯算子图像锐化  
void XiaoP::sharpenImage2(const Mat &image, Mat &result)
{
	Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image, result, image.depth(), kernel);
}

//获得直方图分布图
Mat XiaoP::getHistImg(Mat img)
{

	const int channels[1] = { 0 };
	const int histSize[1] = { 256 };
	float hranges[2] = { 0,255 };
	const float* ranges[1] = { hranges };

	MatND hist;
	calcHist(&img, 1, channels, Mat(), hist, 1, histSize, ranges);

	double maxVal = 0;
	double minVal = 0;

	//找到直方图中的最大值和最小值
	minMaxLoc(hist, &minVal, &maxVal, 0, 0);
	int histimgSize = hist.rows;
	Mat histImg(histimgSize, histimgSize, CV_8U, Scalar(255));
	// 设置最大峰值为图像高度的90%
	int hpt = static_cast<int>(0.9*histimgSize);

	for (int h = 0; h<histimgSize; h++)
	{
		float binVal = hist.at<float>(h);
		int intensity = static_cast<int>(binVal*hpt / maxVal);
		line(histImg, Point(h, histimgSize), Point(h, histimgSize - intensity), Scalar::all(0));
	}

	return histImg;


}

//直方图均衡化
Mat XiaoP::Equalization(Mat img) {
	int N[256];
	float pr[256];
	float dstColor[256];

	for (int i = 0; i<256; i++) {
		N[i] = 0;
		pr[i] = 0.0f;
		dstColor[i] = 0;
	}
	int nr = img.rows;
	int nc = img.cols;


	//统计每个灰度级对应像素个数
	for (int i = 0; i<nr; i++) {
		uchar *data = img.ptr<uchar>(i);
		for (int j = 0; j < nc; j++) {
			int gray = (int)data[j];
			N[gray]++;
		}
	}
	//计算每个灰度级出现概率
	int max = nr*nc;
	for (int i = 0; i<256; i++) {
		pr[i] = N[i] * 1.0f / max;
		if (i>0) {
			dstColor[i] = dstColor[i - 1] + 255 * pr[i];
		}
		else {
			dstColor[i] = 255 * pr[i];
		}
	}
	Mat dst = img.clone();
	//生成均衡后图像 
	for (int i = 0; i<nr; i++) {
		uchar *data = img.ptr<uchar>(i);
		uchar *data1 = dst.ptr<uchar>(i);
		for (int j = 0; j<nc; j++) {
			int temp = (int)data[j];
			int srcColor = temp;
			data1[j] = (int)dstColor[srcColor];
		}
	}
	return dst;
}

//5*5模板通用函数
Mat XiaoP::Templet(Mat img, const int(*templet)[5], int normalize) {
	Mat outimg = img.clone();
	int length = 5;
	int center = (length - 1) / 2;
	int nr = img.rows;
	int nc = img.cols;
	uchar data_add, data_mis, data_add2, data_mis2;
	uchar* temp_z_1, *temp_z_2;
	for (int i = 0; i < nr; i++) {
		uchar* data = img.ptr<uchar>(i);
		uchar* data3 = outimg.ptr<uchar>(i);
		temp_z_1 = data3;
		temp_z_2 = data3;
		for (int j = 0; j < nc; j++) {
			int sum = 0;
			for (int z = 0; z <= center; z++) {
				uchar* data1, *data2;
				if ((i + z) < nr) {
					data1 = outimg.ptr<uchar>(i + z);
					temp_z_1 = data1;
				}
				else {
					data1 = temp_z_1;
				}
				if ((i - z) >= 0) {
					data2 = outimg.ptr<uchar>(i - z);
					temp_z_2 = data2;
				}
				else {
					data2 = temp_z_2;
				}
				for (int x = 0; x <= center; x++) {
					if ((j + x) < nc) {
						data_add = data1[j + x];
						data_add2 = data2[j + x];
					}
					if ((j - x) >= 0) {
						data_mis2 = data2[j - x];
						data_mis = data1[j - x];
					}
					if (z == 0) {
						if (x == 0)
						{
							sum += data_add * templet[center][center + x];
							continue;
						}
						sum += data_add * templet[center][center + x];
						sum += data_mis * templet[center][center - x];
						continue;
					}
					if (x == 0)
					{
						sum += data_add * templet[center + z][center + x];
						sum += data_add2 * templet[center - z][center + x];
						continue;
					}
					sum += data_add * templet[center + z][center + x];
					sum += data_mis * templet[center + z][center - x];
					sum += data_add2 * templet[center - z][center + x];
					sum += data_mis2 * templet[center - z][center - x];
				}
			}
			int temp = sum / normalize;
			data3[j] = temp;
		}
	}
	return outimg;
}

//左边显示图像函数，IplImage格式
void XiaoP::img_display(IplImage* mat)
{
	IplImage* temp = cvCreateImage(cvGetSize(ori_img), IPL_DEPTH_8U, mat->nChannels);
	QImage img;
	if (mat->nChannels == 3)
	{
		cvConvertImage(mat, temp, CV_CVTIMG_SWAP_RB);
		uchar *imgData = (uchar *)temp->imageData;
		img = QImage(imgData, temp->width, temp->height, QImage::Format_RGB888);
	}
	else
	{
		uchar *imgData = (uchar *)mat->imageData;
		img = QImage(imgData, mat->width, mat->height, QImage::Format_Indexed8);
	}
	ui.img->setPixmap(QPixmap::fromImage(img));
	ui.img->resize(ui.img->pixmap()->size());
	ui.img->show();
}


//左边显示图像函数，Mat格式
void XiaoP::img_display(Mat mat)
{
	Mat rgb;
	QImage img;
	if (mat.channels() == 3)
	{
		cv::cvtColor(mat, rgb, CV_BGR2RGB);
		img = QImage((const uchar*)(rgb.data), rgb.cols, rgb.rows, rgb.cols*rgb.channels(), QImage::Format_RGB888);
	}
	else
	{
		img = QImage((const uchar*)(mat.data), mat.cols, mat.rows, mat.cols*mat.channels(), QImage::Format_Indexed8);
	}
	ui.img->setPixmap(QPixmap::fromImage(img));
	ui.img->resize(ui.img->pixmap()->size());
	ui.img->show();
}
//右边显示图像函数，IplImage格式
void XiaoP::dst_display(IplImage* mat)
{
	QImage img;
	if (mat->nChannels == 3)
	{
		cvConvertImage(mat, mat, CV_CVTIMG_SWAP_RB);
		uchar *imgData = (uchar *)mat->imageData;
		img = QImage(imgData, mat->width, mat->height, QImage::Format_RGB888);
	}
	else
	{
		uchar *imgData = (uchar *)mat->imageData;
		img = QImage(imgData, mat->width, mat->height, QImage::Format_Indexed8);
	}
	ui.dst->setPixmap(QPixmap::fromImage(img));
	ui.dst->resize(ui.img->pixmap()->size());
	ui.dst->show();
}


//右边显示图像函数，Mat格式
void XiaoP::dst_display(Mat mat)
{
	Mat rgb;
	QImage img;
	if (mat.channels() == 3)
	{
		cv::cvtColor(mat, rgb, CV_BGR2RGB);
		img = QImage((const uchar*)(rgb.data), rgb.cols, rgb.rows, rgb.cols*rgb.channels(), QImage::Format_RGB888);
	}
	else
	{
		img = QImage((const uchar*)(mat.data), mat.cols, mat.rows, mat.cols*mat.channels(), QImage::Format_Indexed8);
	}
	ui.dst->setPixmap(QPixmap::fromImage(img));
	ui.dst->resize(ui.img->pixmap()->size());
	ui.dst->show();
}
/** 频域滤波函数
* @parameter
* fourier 傅里叶变换图像，二通道
* FlAG 全局变量编辑，确定用哪个滤波器
* d0 滤波器半径
* n1 巴特沃斯滤波器参数
*/
void XiaoP::PassFilter(IplImage * fourier, int FLAG, double d0, int n1)
{
	int i, j;
	int state = -1;
	double tempD;
	long width, height;
	width = fourier->width;
	height = fourier->height;
	long x, y;
	x = width / 2;
	y = height / 2;
	CvMat* H_mat;
	H_mat = cvCreateMat(fourier->height, fourier->width, CV_64FC2);
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			if (i > y && j > x) {
				state = 3;
			}
			else if (i > y) {
				state = 1;
			}
			else if (j > x) {
				state = 2;
			}
			else {
				state = 0;
			}

			switch (state) {
			case 0:
				tempD = (double)sqrt(1.0*i * i + j * j); break;
			case 1:
				tempD = (double)sqrt(1.0*(height - i) * (height - i) + j * j); break;
			case 2:
				tempD = (double)sqrt(1.0*i * i + (width - j) * (width - j)); break;
			case 3:
				tempD = (double)sqrt(1.0*(height - i) * (height - i) + (width - j) * (width - j)); break;
			default:
				break;
			}
			switch (FLAG) {
			case XiaoP::IDEAL_LOW:
				if (tempD <= d0) {
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = 1.0;
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
				}
				else {
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = 0.0;
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
				}
				break;
			case XiaoP::IDEAL_HIGH:
				if (tempD <= d0) {
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = 0.0;
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
				}
				else {
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = 1.0;
					((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
				}
				break;
			case XiaoP::BW_LOW:
				tempD = 1 / (1 + pow(tempD / d0, 2 * n1));
				((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = tempD;
				((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
				break;
			case XiaoP::BW_HIGH:
				tempD = 1 / (1 + pow(d0 / tempD, 2 * n1));
				((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2] = tempD;
				((double*)(H_mat->data.ptr + H_mat->step * i))[j * 2 + 1] = 0.0;
				break;
			default:
				break;
			}
		}
	}
	cvMulSpectrums(fourier, H_mat, fourier, CV_DXT_ROWS);
	cvReleaseMat(&H_mat);
}


IplImage* XiaoP::AddGuassianNoise(IplImage* src)    //添加高斯噪声
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	IplImage* noise = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	CvRNG rng = cvRNG(-1);
	cvRandArr(&rng, noise, CV_RAND_NORMAL, cvScalarAll(0), cvScalarAll(15));
	cvAdd(src, noise, dst);
	return dst;
}
IplImage* XiaoP::AddPepperNoise(IplImage* src)      //添加胡椒噪声，随机黑色点
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvCopy(src, dst);
	for (int k = 0; k<8000; k++)
	{
		int i = rand() % src->height;
		int j = rand() % src->width;
		CvScalar s = cvGet2D(src, i, j);
		if (src->nChannels == 1)
		{
			s.val[0] = 0;
		}
		else if (src->nChannels == 3)
		{
			s.val[0] = 0;
			s.val[1] = 0;
			s.val[2] = 0;
		}
		cvSet2D(dst, i, j, s);
	}
	return dst;
}
IplImage* XiaoP::AddSaltNoise(IplImage* src)       //添加盐噪声，随机白色点
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvCopy(src, dst);
	for (int k = 0; k<8000; k++)
	{
		int i = rand() % src->height;
		int j = rand() % src->width;
		CvScalar s = cvGet2D(src, i, j);
		if (src->nChannels == 1)
		{
			s.val[0] = 255;
		}
		else if (src->nChannels == 3)
		{
			s.val[0] = 255;
			s.val[1] = 255;
			s.val[2] = 255;
		}
		cvSet2D(dst, i, j, s);
	}
	return dst;
}
IplImage* XiaoP::AddPepperSaltNoise(IplImage* src)    //添加椒盐噪声，随机黑白点
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvCopy(src, dst);
	for (int k = 0; k<8000; k++)
	{
		int i = rand() % src->height;
		int j = rand() % src->width;
		int m = rand() % 2;
		CvScalar s = cvGet2D(src, i, j);
		if (src->nChannels == 1)
		{
			if (m == 0)
			{
				s.val[0] = 255;
			}
			else
			{
				s.val[0] = 0;
			}
		}
		else if (src->nChannels == 3)
		{
			if (m == 0)
			{
				s.val[0] = 255;
				s.val[1] = 255;
				s.val[2] = 255;
			}
			else
			{
				s.val[0] = 0;
				s.val[1] = 0;
				s.val[2] = 0;
			}
		}
		cvSet2D(dst, i, j, s);
	}
	return dst;
}

//算术均值滤波器——模板大小5*5
IplImage* XiaoP::ArithmeticMeanFilter(IplImage* src)
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvSmooth(src, dst, CV_BLUR, 5);
	return dst;
}
//几何均值滤波器——模板大小5*5
IplImage* XiaoP::GeometryMeanFilter(IplImage* src)
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	int row, col;
	int h = src->height;
	int w = src->width;
	double mul[3];
	double dc[3];
	int mn;
	//计算每个像素的去噪后color值
	for (int i = 0; i<src->height; i++) {
		for (int j = 0; j<src->width; j++) {
			mul[0] = 1.0;
			mn = 0;
			//统计邻域内的几何平均值，邻域大小5*5
			for (int m = -2; m <= 2; m++) {
				row = i + m;
				for (int n = -2; n <= 2; n++) {
					col = j + n;
					if (row >= 0 && row<h && col >= 0 && col<w) {
						CvScalar s = cvGet2D(src, row, col);
						mul[0] = mul[0] * (s.val[0] == 0 ? 1 : s.val[0]);   //邻域内的非零像素点相乘
						mn++;
					}
				}
			}
			//计算1/mn次方
			CvScalar d;
			dc[0] = pow(mul[0], 1.0 / mn);
			d.val[0] = dc[0];
			//统计成功赋给去噪后图像。
			cvSet2D(dst, i, j, d);
		}
	}
	return dst;
}

//谐波均值滤波器——模板大小5*5
IplImage* XiaoP::HarmonicMeanFilter(IplImage* src)
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	int row, col;
	int h = src->height;
	int w = src->width;
	double sum[3];
	double dc[3];
	int mn;
	//计算每个像素的去噪后color值
	for (int i = 0; i<src->height; i++) {
		for (int j = 0; j<src->width; j++) {
			sum[0] = 0.0;
			mn = 0;
			//统计邻域,5*5模板
			for (int m = -2; m <= 2; m++) {
				row = i + m;
				for (int n = -2; n <= 2; n++) {
					col = j + n;
					if (row >= 0 && row<h && col >= 0 && col<w) {
						CvScalar s = cvGet2D(src, row, col);
						sum[0] = sum[0] + (s.val[0] == 0 ? 255 : 255 / s.val[0]);
						mn++;
					}
				}
			}
			CvScalar d;
			dc[0] = mn * 255 / sum[0];
			d.val[0] = dc[0];
			//统计成功赋给去噪后图像。
			cvSet2D(dst, i, j, d);
		}
	}
	return dst;
}
//逆谐波均值大小滤波器——模板大小5*5
IplImage* XiaoP::InverseHarmonicMeanFilter(IplImage* src)
{
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	//cvSmooth(src,dst,CV_BLUR,5); 
	int row, col;
	int h = src->height;
	int w = src->width;
	double sum[3];
	double sum1[3];
	double dc[3];
	double Q = 2;
	//计算每个像素的去噪后color值
	for (int i = 0; i<src->height; i++) {
		for (int j = 0; j<src->width; j++) {
			sum[0] = 0.0;
			sum1[0] = 0.0;
			//统计邻域
			for (int m = -2; m <= 2; m++) {
				row = i + m;
				for (int n = -2; n <= 2; n++) {
					col = j + n;
					if (row >= 0 && row<h && col >= 0 && col<w) {
						CvScalar s = cvGet2D(src, row, col);
						sum[0] = sum[0] + pow(s.val[0] / 255, Q + 1);
						sum1[0] = sum1[0] + pow(s.val[0] / 255, Q);
					}
				}
			}
			//计算1/mn次方
			CvScalar d;
			dc[0] = (sum1[0] == 0 ? 0 : (sum[0] / sum1[0])) * 255;
			d.val[0] = dc[0];
			//统计成功赋给去噪后图像。
			cvSet2D(dst, i, j, d);
		}
	}
	return dst;
}
//中值滤波 模板大小5*5
IplImage* XiaoP::MedianFilter_5_5(IplImage* src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvSmooth(src, dst, CV_MEDIAN, 5);
	return dst;
}
//中值滤波 模板大小9*9
IplImage* XiaoP::MedianFilter_9_9(IplImage* src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvSmooth(src, dst, CV_MEDIAN, 9);
	return dst;
}
//自适应均值滤波
IplImage* XiaoP::SelfAdaptMeanFilter(IplImage* src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	cvSmooth(src, dst, CV_BLUR, 7);
	int row, col;
	int h = src->height;
	int w = src->width;
	int mn;
	double Zxy;
	double Zmed;
	double Sxy;
	double Sl;
	double Sn = 100;
	for (int i = 0; i<src->height; i++) {
		for (int j = 0; j<src->width; j++) {
			CvScalar xy = cvGet2D(src, i, j);
			Zxy = xy.val[0];
			CvScalar dxy = cvGet2D(dst, i, j);
			Zmed = dxy.val[0];
			Sl = 0;
			mn = 0;
			for (int m = -3; m <= 3; m++) {
				row = i + m;
				for (int n = -3; n <= 3; n++) {
					col = j + n;
					if (row >= 0 && row<h && col >= 0 && col<w) {
						CvScalar s = cvGet2D(src, row, col);
						Sxy = s.val[0];
						Sl = Sl + pow(Sxy - Zmed, 2);
						mn++;
					}
				}
			}
			Sl = Sl / mn;
			CvScalar d;
			d.val[0] = Zxy - Sn / Sl*(Zxy - Zmed);
			cvSet2D(dst, i, j, d);
		}
	}
	return dst;
}
//自适应中值滤波
IplImage* XiaoP::SelfAdaptMedianFilter(IplImage* src) {
	IplImage* dst = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
	int row, col;
	int h = src->height;
	int w = src->width;
	double Zmin, Zmax, Zmed, Zxy, Smax = 7;
	int wsize;
	//计算每个像素的去噪后color值
	for (int i = 0; i<src->height; i++) {
		for (int j = 0; j<src->width; j++) {
			//统计邻域
			wsize = 1;
			while (wsize <= 3) {
				Zmin = 255.0;
				Zmax = 0.0;
				Zmed = 0.0;
				CvScalar xy = cvGet2D(src, i, j);
				Zxy = xy.val[0];
				int mn = 0;
				for (int m = -wsize; m <= wsize; m++) {
					row = i + m;
					for (int n = -wsize; n <= wsize; n++) {
						col = j + n;
						if (row >= 0 && row<h && col >= 0 && col<w) {
							CvScalar s = cvGet2D(src, row, col);
							if (s.val[0]>Zmax) {
								Zmax = s.val[0];
							}
							if (s.val[0]<Zmin) {
								Zmin = s.val[0];
							}
							Zmed = Zmed + s.val[0];
							mn++;
						}
					}
				}
				Zmed = Zmed / mn;
				CvScalar d;
				if ((Zmed - Zmin)>0 && (Zmed - Zmax)<0) {
					if ((Zxy - Zmin)>0 && (Zxy - Zmax)<0) {
						d.val[0] = Zxy;
					}
					else {
						d.val[0] = Zmed;
					}
					cvSet2D(dst, i, j, d);
					break;
				}
				else {
					wsize++;
					if (wsize>3) {
						CvScalar d;
						d.val[0] = Zmed;
						cvSet2D(dst, i, j, d);
						break;
					}
				}
			}
		}
	}
	return dst;
}

//DFT变换
IplImage* XiaoP::DFT(IplImage * src)
{
	IplImage* fourier = cvCreateImage(cvGetSize(src), IPL_DEPTH_64F, 2);
	int dft_H, dft_W;
	dft_H = src->height;
	dft_W = src->width;
	CvMat *src_Re = cvCreateMat(dft_H, dft_W, CV_64FC1);	// double Re, Im;
	CvMat *src_Im = cvCreateMat(dft_H, dft_W, CV_64FC1);	//Imaginary part	
	CvMat *sum_src = cvCreateMat(dft_H, dft_W, CV_64FC2);	//2 channels (src_Re, src_Im)
	CvMat *sum_dst = cvCreateMat(dft_H, dft_W, CV_64FC2);	//2 channels (dst_Re, dst_Im)
	cvConvert(src, src_Re);
	cvZero(src_Im);
	cvMerge(src_Re, src_Im, 0, 0, sum_src);
	cvDFT(sum_src, sum_dst, CV_DXT_FORWARD, 0);
	cvConvert(sum_dst, fourier);
	cvReleaseMat(&src_Re);
	cvReleaseMat(&src_Im);
	cvReleaseMat(&sum_src);
	cvReleaseMat(&sum_dst);
	return fourier;
}
//DFT反变换
IplImage* XiaoP::IDFT(IplImage * fourier)
{
	IplImage* dst = cvCreateImage(cvGetSize(fourier), IPL_DEPTH_8U, 1);
	int dft_H, dft_W;
	dft_H = fourier->height;
	dft_W = fourier->width;
	CvMat *dst_Re = cvCreateMat(dft_H, dft_W, CV_64FC1);	// double Re, Im;
	CvMat *dst_Im = cvCreateMat(dft_H, dft_W, CV_64FC1);	//Imaginary part
	CvMat *sum_dst = cvCreateMat(dft_H, dft_W, CV_64FC2);	//2 channels (dst_Re, dst_Im)
	CvMat *sum_src = cvCreateMat(dft_H, dft_W, CV_64FC2);
	cvConvert(fourier, sum_src);
	cvDFT(sum_src, sum_dst, CV_DXT_INV_SCALE, 0);
	cvSplit(sum_dst, dst_Re, dst_Im, 0, 0);
	cvConvert(dst_Re, dst);
	cvReleaseMat(&dst_Re);
	cvReleaseMat(&dst_Im);
	cvReleaseMat(&sum_src);
	cvReleaseMat(&sum_dst);
	return dst;
}

//归一化，将灰度映射到0~255之间, 并将能量最高的四角移到中心, 生成图片频域能量图
void XiaoP::BuildDFTImage(IplImage *fourier, IplImage *dst)
{
	IplImage *image_Re = 0, *image_Im = 0;

	image_Re = cvCreateImage(cvGetSize(fourier), IPL_DEPTH_64F, 1);
	image_Im = cvCreateImage(cvGetSize(fourier), IPL_DEPTH_64F, 1);	//Imaginary part

	cvSplit(fourier, image_Re, image_Im, 0, 0);

	// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)

	cvPow(image_Re, image_Re, 2.0);
	cvPow(image_Im, image_Im, 2.0);

	cvAdd(image_Re, image_Im, image_Re);
	cvPow(image_Re, image_Re, 0.5);

	cvReleaseImage(&image_Im);

	cvAddS(image_Re, cvScalar(1.0), image_Re); // 1 + Mag
	cvLog(image_Re, image_Re); // log(1 + Mag)

							   //重新安排傅里叶图像中心
							   // Rearrange the quadrants of Fourier image so that the origin is at
							   // the image center
	double minVal = 0, maxVal = 0;
	cvMinMaxLoc(image_Re, &minVal, &maxVal);	// Localize minimum and maximum values

	CvScalar min;
	min.val[0] = minVal;
	double scale = 255 / (maxVal - minVal);

	cvSubS(image_Re, min, image_Re);
	cvConvertScale(image_Re, dst, scale);
	cvReleaseImage(&image_Re);

	// Rearrange the quadrants of Fourier image so that the origin is at
	// the image center
	int nRow, nCol, i, j, cy, cx;
	uchar tmp13, tmp24;

	nRow = fourier->height;
	nCol = fourier->width;
	cy = nRow / 2; // image center
	cx = nCol / 2;
	for (j = 0; j < cy; j++)
	{
		for (i = 0; i < cx; i++)
		{
			tmp13 = CV_IMAGE_ELEM(dst, uchar, j, i);
			CV_IMAGE_ELEM(dst, uchar, j, i) = CV_IMAGE_ELEM(dst, uchar, j + cy, i + cx);
			CV_IMAGE_ELEM(dst, uchar, j + cy, i + cx) = tmp13;

			tmp24 = CV_IMAGE_ELEM(dst, uchar, j, i + cx);
			CV_IMAGE_ELEM(dst, uchar, j, i + cx) = CV_IMAGE_ELEM(dst, uchar, j + cy, i);
			CV_IMAGE_ELEM(dst, uchar, j + cy, i) = tmp24;
		}
	}
}