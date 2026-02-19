/********************************************************************************
** Form generated from reading UI file 'maineyZjlg.ui'
**
** Created by: Qt User Interface Compiler version 6.7.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef MAINEYZJLG_H
#define MAINEYZJLG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QIcon>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen;
    QAction *actionExport;
    QAction *actionCommand_Bar;
    QAction *actionCNC_Editor;
    QAction *action_move_down;
    QAction *action_move_up;
    QAction *action_single_sewing;
    QAction *action_parallel_sewing;
    QAction *action_mirror_sewing;
    QAction *action_delete;
    QAction *actionRedo;
    QAction *actionUndo;
    QWidget *centralwidget;
    QGridLayout *gridLayout;
    QMenuBar *menubar;
    QMenu *menuFile;
    QMenu *menuEdit;
    QStatusBar *statusbar;
    QDockWidget *CNC_EDITOR;
    QWidget *dockWidgetContents;
    QGridLayout *gridLayout_4;
    QGridLayout *gridLayout_3;
    QListWidget *listWidget;
    QPushButton *pushButton_2;
    QPushButton *pushButton;
    QToolBar *toolBar;
    QToolBar *toolBar_2;
    QDockWidget *dockWidget_2;
    QWidget *dockWidgetContents_4;
    QGridLayout *gridLayout_8;
    QGridLayout *gridLayout_7;
    QWidget *widget;
    QGridLayout *gridLayout_9;
    QLabel *label;
    QSlider *horizontalSlider;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(946, 512);
        actionOpen = new QAction(MainWindow);
        actionOpen->setObjectName("actionOpen");
        actionExport = new QAction(MainWindow);
        actionExport->setObjectName("actionExport");
        actionCommand_Bar = new QAction(MainWindow);
        actionCommand_Bar->setObjectName("actionCommand_Bar");
        actionCommand_Bar->setCheckable(true);
        actionCommand_Bar->setChecked(true);
        actionCNC_Editor = new QAction(MainWindow);
        actionCNC_Editor->setObjectName("actionCNC_Editor");
        actionCNC_Editor->setCheckable(true);
        actionCNC_Editor->setChecked(true);
        action_move_down = new QAction(MainWindow);
        action_move_down->setObjectName("action_move_down");
        QIcon icon;
        icon.addFile(QString::fromUtf8("../../../Downloads/down-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        action_move_down->setIcon(icon);
        action_move_down->setMenuRole(QAction::MenuRole::NoRole);
        action_move_up = new QAction(MainWindow);
        action_move_up->setObjectName("action_move_up");
        QIcon icon1;
        icon1.addFile(QString::fromUtf8("../../../Downloads/up-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::On);
        action_move_up->setIcon(icon1);
        action_move_up->setMenuRole(QAction::MenuRole::NoRole);
        action_single_sewing = new QAction(MainWindow);
        action_single_sewing->setObjectName("action_single_sewing");
        QIcon icon2;
        icon2.addFile(QString::fromUtf8("../../../Downloads/one-pencil-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        icon2.addFile(QString::fromUtf8("../../../Downloads/up-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::On);
        action_single_sewing->setIcon(icon2);
        action_single_sewing->setMenuRole(QAction::MenuRole::NoRole);
        action_parallel_sewing = new QAction(MainWindow);
        action_parallel_sewing->setObjectName("action_parallel_sewing");
        QIcon icon3;
        icon3.addFile(QString::fromUtf8("../../../Downloads/two-pencil-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        icon3.addFile(QString::fromUtf8("../../../Downloads/up-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::On);
        action_parallel_sewing->setIcon(icon3);
        action_parallel_sewing->setMenuRole(QAction::MenuRole::NoRole);
        action_mirror_sewing = new QAction(MainWindow);
        action_mirror_sewing->setObjectName("action_mirror_sewing");
        QIcon icon4;
        icon4.addFile(QString::fromUtf8("../../../Downloads/mirror-pencil-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        icon4.addFile(QString::fromUtf8("../../../Downloads/up-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::On);
        action_mirror_sewing->setIcon(icon4);
        action_mirror_sewing->setMenuRole(QAction::MenuRole::NoRole);
        action_delete = new QAction(MainWindow);
        action_delete->setObjectName("action_delete");
        QIcon icon5;
        icon5.addFile(QString::fromUtf8("../../../Downloads/delete-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        icon5.addFile(QString::fromUtf8("../../../Downloads/up-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::On);
        action_delete->setIcon(icon5);
        action_delete->setMenuRole(QAction::MenuRole::NoRole);
        actionRedo = new QAction(MainWindow);
        actionRedo->setObjectName("actionRedo");
        actionUndo = new QAction(MainWindow);
        actionUndo->setObjectName("actionUndo");
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        gridLayout = new QGridLayout(centralwidget);
        gridLayout->setObjectName("gridLayout");
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 946, 33));
        menuFile = new QMenu(menubar);
        menuFile->setObjectName("menuFile");
        menuEdit = new QMenu(menubar);
        menuEdit->setObjectName("menuEdit");
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName("statusbar");
        MainWindow->setStatusBar(statusbar);
        CNC_EDITOR = new QDockWidget(MainWindow);
        CNC_EDITOR->setObjectName("CNC_EDITOR");
        CNC_EDITOR->setStyleSheet(QString::fromUtf8("QDockWidget > QWidget {\n"
"    border: 0.5px solid black;\n"
"}"));
        CNC_EDITOR->setFeatures(QDockWidget::DockWidgetFeature::DockWidgetMovable);
        dockWidgetContents = new QWidget();
        dockWidgetContents->setObjectName("dockWidgetContents");
        gridLayout_4 = new QGridLayout(dockWidgetContents);
        gridLayout_4->setObjectName("gridLayout_4");
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setObjectName("gridLayout_3");
        listWidget = new QListWidget(dockWidgetContents);
        listWidget->setObjectName("listWidget");
        listWidget->setAcceptDrops(true);

        gridLayout_3->addWidget(listWidget, 1, 0, 1, 1);

        pushButton_2 = new QPushButton(dockWidgetContents);
        pushButton_2->setObjectName("pushButton_2");
        pushButton_2->setMinimumSize(QSize(0, 40));
        QIcon icon6;
        icon6.addFile(QString::fromUtf8("../../../Downloads/export-content-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        pushButton_2->setIcon(icon6);
        pushButton_2->setIconSize(QSize(22, 22));

        gridLayout_3->addWidget(pushButton_2, 2, 0, 1, 1);

        pushButton = new QPushButton(dockWidgetContents);
        pushButton->setObjectName("pushButton");
        pushButton->setMinimumSize(QSize(0, 36));
        QIcon icon7;
        icon7.addFile(QString::fromUtf8("../../../Downloads/drag-right-svgrepo-com.svg"), QSize(), QIcon::Mode::Normal, QIcon::State::Off);
        pushButton->setIcon(icon7);
        pushButton->setIconSize(QSize(20, 20));

        gridLayout_3->addWidget(pushButton, 0, 0, 1, 1);


        gridLayout_4->addLayout(gridLayout_3, 0, 0, 1, 1);

        CNC_EDITOR->setWidget(dockWidgetContents);
        MainWindow->addDockWidget(Qt::DockWidgetArea::LeftDockWidgetArea, CNC_EDITOR);
        toolBar = new QToolBar(MainWindow);
        toolBar->setObjectName("toolBar");
        MainWindow->addToolBar(Qt::ToolBarArea::LeftToolBarArea, toolBar);
        toolBar_2 = new QToolBar(MainWindow);
        toolBar_2->setObjectName("toolBar_2");
        MainWindow->addToolBar(Qt::ToolBarArea::LeftToolBarArea, toolBar_2);
        dockWidget_2 = new QDockWidget(MainWindow);
        dockWidget_2->setObjectName("dockWidget_2");
        QSizePolicy sizePolicy(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(dockWidget_2->sizePolicy().hasHeightForWidth());
        dockWidget_2->setSizePolicy(sizePolicy);
        dockWidget_2->setStyleSheet(QString::fromUtf8("QDockWidget > QWidget {\n"
"    border: 0.5px solid black;\n"
"}"));
        dockWidgetContents_4 = new QWidget();
        dockWidgetContents_4->setObjectName("dockWidgetContents_4");
        gridLayout_8 = new QGridLayout(dockWidgetContents_4);
        gridLayout_8->setObjectName("gridLayout_8");
        gridLayout_7 = new QGridLayout();
        gridLayout_7->setObjectName("gridLayout_7");
        widget = new QWidget(dockWidgetContents_4);
        widget->setObjectName("widget");
        sizePolicy.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy);
        gridLayout_9 = new QGridLayout(widget);
        gridLayout_9->setObjectName("gridLayout_9");
        label = new QLabel(widget);
        label->setObjectName("label");
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);

        gridLayout_9->addWidget(label, 0, 0, 1, 1);


        gridLayout_7->addWidget(widget, 0, 0, 1, 1);

        horizontalSlider = new QSlider(dockWidgetContents_4);
        horizontalSlider->setObjectName("horizontalSlider");
        QSizePolicy sizePolicy1(QSizePolicy::Policy::Expanding, QSizePolicy::Policy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(horizontalSlider->sizePolicy().hasHeightForWidth());
        horizontalSlider->setSizePolicy(sizePolicy1);
        horizontalSlider->setOrientation(Qt::Orientation::Horizontal);

        gridLayout_7->addWidget(horizontalSlider, 1, 0, 1, 1);


        gridLayout_8->addLayout(gridLayout_7, 0, 0, 1, 1);

        dockWidget_2->setWidget(dockWidgetContents_4);
        MainWindow->addDockWidget(Qt::DockWidgetArea::RightDockWidgetArea, dockWidget_2);

        menubar->addAction(menuFile->menuAction());
        menubar->addAction(menuEdit->menuAction());
        menuFile->addAction(actionOpen);
        menuFile->addAction(actionExport);
        menuEdit->addAction(actionRedo);
        menuEdit->addAction(actionUndo);
        toolBar->addSeparator();
        toolBar->addAction(action_move_up);
        toolBar->addAction(action_move_down);
        toolBar->addSeparator();
        toolBar->addAction(action_delete);
        toolBar->addSeparator();
        toolBar_2->addAction(action_single_sewing);
        toolBar_2->addAction(action_parallel_sewing);
        toolBar_2->addAction(action_mirror_sewing);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "CNC Workspace", nullptr));
        actionOpen->setText(QCoreApplication::translate("MainWindow", "Open", nullptr));
        actionExport->setText(QCoreApplication::translate("MainWindow", "Export", nullptr));
        actionCommand_Bar->setText(QCoreApplication::translate("MainWindow", "Tools", nullptr));
        actionCNC_Editor->setText(QCoreApplication::translate("MainWindow", "CNC Edit", nullptr));
        action_move_down->setText(QCoreApplication::translate("MainWindow", "move_down", nullptr));
#if QT_CONFIG(tooltip)
        action_move_down->setToolTip(QCoreApplication::translate("MainWindow", "Move Down", nullptr));
#endif // QT_CONFIG(tooltip)
        action_move_up->setText(QCoreApplication::translate("MainWindow", "move_up", nullptr));
#if QT_CONFIG(tooltip)
        action_move_up->setToolTip(QCoreApplication::translate("MainWindow", "Move Up", nullptr));
#endif // QT_CONFIG(tooltip)
        action_single_sewing->setText(QCoreApplication::translate("MainWindow", "convert_to_single_head", nullptr));
#if QT_CONFIG(tooltip)
        action_single_sewing->setToolTip(QCoreApplication::translate("MainWindow", "Single Head Sewing For Selected Objects", nullptr));
#endif // QT_CONFIG(tooltip)
        action_parallel_sewing->setText(QCoreApplication::translate("MainWindow", "convert_to_dual_head_parallel", nullptr));
#if QT_CONFIG(tooltip)
        action_parallel_sewing->setToolTip(QCoreApplication::translate("MainWindow", "Dual Head Parallel Sewing For Selected Objects", nullptr));
#endif // QT_CONFIG(tooltip)
        action_mirror_sewing->setText(QCoreApplication::translate("MainWindow", "convert_to_dual_head_mirror", nullptr));
#if QT_CONFIG(tooltip)
        action_mirror_sewing->setToolTip(QCoreApplication::translate("MainWindow", "Dual Head Mirror Sewing For Selected Objects", nullptr));
#endif // QT_CONFIG(tooltip)
        action_delete->setText(QCoreApplication::translate("MainWindow", "delete", nullptr));
#if QT_CONFIG(tooltip)
        action_delete->setToolTip(QCoreApplication::translate("MainWindow", "Delete", nullptr));
#endif // QT_CONFIG(tooltip)
        actionRedo->setText(QCoreApplication::translate("MainWindow", "Undo", nullptr));
        actionUndo->setText(QCoreApplication::translate("MainWindow", "Redo", nullptr));
        menuFile->setTitle(QCoreApplication::translate("MainWindow", "File", nullptr));
        menuEdit->setTitle(QCoreApplication::translate("MainWindow", "Edit", nullptr));
        CNC_EDITOR->setWindowTitle(QCoreApplication::translate("MainWindow", "CNC Editor", nullptr));
        pushButton_2->setText(QCoreApplication::translate("MainWindow", " Export CNC", nullptr));
        pushButton->setText(QCoreApplication::translate("MainWindow", " Import Design (DXF, CNC)", nullptr));
        toolBar->setWindowTitle(QCoreApplication::translate("MainWindow", "toolBar", nullptr));
        toolBar_2->setWindowTitle(QCoreApplication::translate("MainWindow", "toolBar_2", nullptr));
        dockWidget_2->setWindowTitle(QCoreApplication::translate("MainWindow", "CNC Visualization", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "TextLabel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // MAINEYZJLG_H
