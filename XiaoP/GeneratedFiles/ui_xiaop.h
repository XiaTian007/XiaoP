/********************************************************************************
** Form generated from reading UI file 'xiaop.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_XIAOP_H
#define UI_XIAOP_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_XiaoPClass
{
public:
    QWidget *centralWidget;
    QPushButton *openButton;
    QLabel *img;
    QLabel *dst;
    QPushButton *grayButton;
    QPushButton *histButton;
    QPushButton *histEqualButton;
    QPushButton *oriButton;
    QPushButton *equalButton;
    QPushButton *medianButton;
    QPushButton *gaussButton;
    QPushButton *saltButton;
    QPushButton *pepperButton;
    QPushButton *pepperSaltButton;
    QPushButton *gaussNoiseButton;
    QPushButton *sharpButton;
    QPushButton *DFTButton;
    QPushButton *LPButton;
    QPushButton *laplaceButton;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *XiaoPClass)
    {
        if (XiaoPClass->objectName().isEmpty())
            XiaoPClass->setObjectName(QStringLiteral("XiaoPClass"));
        XiaoPClass->resize(1002, 632);
        centralWidget = new QWidget(XiaoPClass);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        openButton = new QPushButton(centralWidget);
        openButton->setObjectName(QStringLiteral("openButton"));
        openButton->setGeometry(QRect(60, 480, 81, 41));
        img = new QLabel(centralWidget);
        img->setObjectName(QStringLiteral("img"));
        img->setGeometry(QRect(20, 20, 441, 441));
        dst = new QLabel(centralWidget);
        dst->setObjectName(QStringLiteral("dst"));
        dst->setGeometry(QRect(520, 20, 441, 441));
        grayButton = new QPushButton(centralWidget);
        grayButton->setObjectName(QStringLiteral("grayButton"));
        grayButton->setGeometry(QRect(160, 480, 81, 41));
        histButton = new QPushButton(centralWidget);
        histButton->setObjectName(QStringLiteral("histButton"));
        histButton->setGeometry(QRect(500, 480, 81, 41));
        histEqualButton = new QPushButton(centralWidget);
        histEqualButton->setObjectName(QStringLiteral("histEqualButton"));
        histEqualButton->setGeometry(QRect(600, 530, 81, 41));
        oriButton = new QPushButton(centralWidget);
        oriButton->setObjectName(QStringLiteral("oriButton"));
        oriButton->setGeometry(QRect(60, 530, 81, 41));
        equalButton = new QPushButton(centralWidget);
        equalButton->setObjectName(QStringLiteral("equalButton"));
        equalButton->setGeometry(QRect(700, 480, 81, 41));
        medianButton = new QPushButton(centralWidget);
        medianButton->setObjectName(QStringLiteral("medianButton"));
        medianButton->setGeometry(QRect(600, 480, 81, 41));
        gaussButton = new QPushButton(centralWidget);
        gaussButton->setObjectName(QStringLiteral("gaussButton"));
        gaussButton->setGeometry(QRect(800, 480, 81, 41));
        saltButton = new QPushButton(centralWidget);
        saltButton->setObjectName(QStringLiteral("saltButton"));
        saltButton->setGeometry(QRect(210, 530, 81, 41));
        pepperButton = new QPushButton(centralWidget);
        pepperButton->setObjectName(QStringLiteral("pepperButton"));
        pepperButton->setGeometry(QRect(260, 480, 81, 41));
        pepperSaltButton = new QPushButton(centralWidget);
        pepperSaltButton->setObjectName(QStringLiteral("pepperSaltButton"));
        pepperSaltButton->setGeometry(QRect(360, 480, 81, 41));
        gaussNoiseButton = new QPushButton(centralWidget);
        gaussNoiseButton->setObjectName(QStringLiteral("gaussNoiseButton"));
        gaussNoiseButton->setGeometry(QRect(360, 530, 81, 41));
        sharpButton = new QPushButton(centralWidget);
        sharpButton->setObjectName(QStringLiteral("sharpButton"));
        sharpButton->setGeometry(QRect(800, 530, 81, 41));
        DFTButton = new QPushButton(centralWidget);
        DFTButton->setObjectName(QStringLiteral("DFTButton"));
        DFTButton->setGeometry(QRect(900, 530, 81, 41));
        LPButton = new QPushButton(centralWidget);
        LPButton->setObjectName(QStringLiteral("LPButton"));
        LPButton->setGeometry(QRect(900, 480, 81, 41));
        laplaceButton = new QPushButton(centralWidget);
        laplaceButton->setObjectName(QStringLiteral("laplaceButton"));
        laplaceButton->setGeometry(QRect(700, 530, 81, 41));
        XiaoPClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(XiaoPClass);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1002, 21));
        XiaoPClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(XiaoPClass);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        XiaoPClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(XiaoPClass);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        XiaoPClass->setStatusBar(statusBar);

        retranslateUi(XiaoPClass);
        QObject::connect(openButton, SIGNAL(clicked()), img, SLOT(repaint()));
        QObject::connect(saltButton, SIGNAL(clicked()), img, SLOT(repaint()));
        QObject::connect(pepperButton, SIGNAL(clicked()), img, SLOT(repaint()));
        QObject::connect(pepperSaltButton, SIGNAL(clicked()), img, SLOT(repaint()));
        QObject::connect(oriButton, SIGNAL(clicked()), img, SLOT(repaint()));
        QObject::connect(gaussNoiseButton, SIGNAL(clicked()), img, SLOT(repaint()));
        QObject::connect(medianButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(equalButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(gaussButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(histButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(histEqualButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(laplaceButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(sharpButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(DFTButton, SIGNAL(clicked()), dst, SLOT(repaint()));
        QObject::connect(grayButton, SIGNAL(clicked()), img, SLOT(repaint()));
        QObject::connect(LPButton, SIGNAL(clicked()), dst, SLOT(repaint()));

        QMetaObject::connectSlotsByName(XiaoPClass);
    } // setupUi

    void retranslateUi(QMainWindow *XiaoPClass)
    {
        XiaoPClass->setWindowTitle(QApplication::translate("XiaoPClass", "XiaoP", Q_NULLPTR));
        openButton->setText(QApplication::translate("XiaoPClass", "\346\211\223\345\274\200\345\233\276\345\203\217", Q_NULLPTR));
        img->setText(QString());
        dst->setText(QString());
        grayButton->setText(QApplication::translate("XiaoPClass", "\347\201\260\345\272\246\345\233\276", Q_NULLPTR));
        histButton->setText(QApplication::translate("XiaoPClass", "\346\230\276\347\244\272\347\233\264\346\226\271\345\233\276", Q_NULLPTR));
        histEqualButton->setText(QApplication::translate("XiaoPClass", "\347\233\264\346\226\271\345\233\276\345\235\207\350\241\241\345\214\226", Q_NULLPTR));
        oriButton->setText(QApplication::translate("XiaoPClass", "\346\201\242\345\244\215\345\216\237\345\233\276", Q_NULLPTR));
        equalButton->setText(QApplication::translate("XiaoPClass", "\345\235\207\345\200\274\346\273\244\346\263\242", Q_NULLPTR));
        medianButton->setText(QApplication::translate("XiaoPClass", "\344\270\255\345\200\274\346\273\244\346\263\242", Q_NULLPTR));
        gaussButton->setText(QApplication::translate("XiaoPClass", "\351\253\230\346\226\257\346\273\244\346\263\242", Q_NULLPTR));
        saltButton->setText(QApplication::translate("XiaoPClass", "\347\233\220\345\231\252\345\243\260", Q_NULLPTR));
        pepperButton->setText(QApplication::translate("XiaoPClass", "\350\203\241\346\244\222\345\231\252\345\243\260", Q_NULLPTR));
        pepperSaltButton->setText(QApplication::translate("XiaoPClass", "\346\244\222\347\233\220\345\231\252\345\243\260", Q_NULLPTR));
        gaussNoiseButton->setText(QApplication::translate("XiaoPClass", "\351\253\230\346\226\257\345\231\252\345\243\260", Q_NULLPTR));
        sharpButton->setText(QApplication::translate("XiaoPClass", "\351\224\220\345\214\226", Q_NULLPTR));
        DFTButton->setText(QApplication::translate("XiaoPClass", "\351\242\221\345\237\237\345\217\230\346\215\242", Q_NULLPTR));
        LPButton->setText(QApplication::translate("XiaoPClass", "\351\242\221\345\237\237\346\273\244\346\263\242", Q_NULLPTR));
        laplaceButton->setText(QApplication::translate("XiaoPClass", "\347\273\206\350\212\202\345\233\276", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class XiaoPClass: public Ui_XiaoPClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_XIAOP_H
