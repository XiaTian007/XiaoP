#include "xiaop.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	XiaoP w;
	w.show();
	return a.exec();
}
