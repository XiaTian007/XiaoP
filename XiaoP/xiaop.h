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
	//�����ͨ
	static const int IDEAL_LOW = 1;
	//�����ͨ
	static const int IDEAL_HIGH = 2;
	//������˹��ͨ
	static const int BW_LOW = 3;
	//������˹��ͨ
	static const int BW_HIGH = 4;
	XiaoP(QWidget *parent = 0);
	~XiaoP();
	//��һ�������Ҷ�ӳ�䵽0~255֮��, ����������ߵ��Ľ��Ƶ�����, ����ͼƬƵ������ͼ
	void BuildDFTImage(IplImage *fourier, IplImage *dst);
	//DFT�任
	IplImage *DFT(IplImage * src);
	//���ξ�ֵ�˲�������ģ���С5*5
	IplImage* GeometryMeanFilter(IplImage* src);
	//RGBToGray
	IplImage* XiaoP::Gray(IplImage* src);
	//г����ֵ�˲�������ģ���С5*5
	IplImage* HarmonicMeanFilter(IplImage* src);
	//DFT���任
	IplImage *IDFT(IplImage * fourier);
	//��г����ֵ��С�˲�������ģ���С5*5
	IplImage* InverseHarmonicMeanFilter(IplImage* src);
	//��ֵ�˲� ģ���С5*5
	IplImage* MedianFilter_5_5(IplImage* src);
	//��ֵ�˲� ģ���С9*9
	IplImage* MedianFilter_9_9(IplImage* src);

	/** Ƶ���˲�����
	* @parameter
	* fourier ����Ҷ�任ͼ�񣬶�ͨ��
	* FlAG ȫ�ֱ����༭��ȷ�����ĸ��˲���
	* d0 �˲����뾶
	* n1 ������˹�˲�������
	*/
	void PassFilter(IplImage * fourier, int FLAG, double d0, int n1);

	//Roberts����ʵ��
	Mat roberts(Mat srcImage);
	//����Ӧ��ֵ�˲�
	IplImage* SelfAdaptMeanFilter(IplImage* src);
	//����Ӧ��ֵ�˲�
	IplImage* SelfAdaptMedianFilter(IplImage* src);
	//�ֶ�ʵ��������˹����ͼ����  
	void sharpenImage1(const Mat &image, Mat &result);
	//����OpenCV����ʵ��������˹����ͼ����  
	void sharpenImage2(const Mat &image, Mat &result);
	//5*5ģ��ͨ�ú���
	Mat Templet(Mat img, const int(*templet)[5], int normalize);

	//���ֱ��ͼ�ֲ�ͼ
	Mat getHistImg(Mat img);
	//ֱ��ͼ���⻯
	Mat Equalization(Mat img);

	IplImage* AddGuassianNoise(IplImage* src);    //��Ӹ�˹����
	IplImage* AddPepperNoise(IplImage* src);      //��Ӻ��������������ɫ��
	IplImage* AddPepperSaltNoise(IplImage* src);    //��ӽ�������������ڰ׵�
	IplImage* AddSaltNoise(IplImage* src);       //����������������ɫ��
	IplImage* ArithmeticMeanFilter(IplImage* src);//������ֵ�˲�������ģ���С5*5
	void img_display(cv::Mat mat);//�ұ���ʾͼ������Mat��ʽ
	void img_display(IplImage* mat);//�ұ���ʾͼ������IplImage��ʽ
	void dst_display(cv::Mat mat);//�����ʾͼ������Mat��ʽ
	void dst_display(IplImage* mat);//�����ʾͼ������IplImage��ʽ
	
private slots:
	void on_openButton_clicked();//��ͼ��ť����
	void on_grayButton_clicked();//��ɫͼ��תΪ�Ҷ�ͼ
	void on_histButton_clicked();//��ʾֱ��ͼ
	void on_histEqualButton_clicked();//ֱ��ͼ����
	void on_oriButton_clicked();//��ʾԭͼ
	void on_equalButton_clicked();//��ֵ�˲�
	void on_medianButton_clicked();//��ֵ�˲�
	void on_gaussButton_clicked();//��˹�˲�
	void on_saltButton_clicked();//��������
	void on_pepperButton_clicked();//��������
	void on_pepperSaltButton_clicked();//��������
	void on_gaussNoiseButton_clicked();//��˹����
	void on_sharpButton_clicked();//��ͼ��
	void on_DFTButton_clicked();//DFT�任
	void on_LPButton_clicked();//��ͨ�˲�
	void on_laplaceButton_clicked();//������˹ͼ

private:

	Ui::XiaoPClass ui;
};

#endif // XIAOP_H
