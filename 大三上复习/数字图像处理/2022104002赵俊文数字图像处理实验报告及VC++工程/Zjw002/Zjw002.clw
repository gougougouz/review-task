; CLW file contains information for the MFC ClassWizard

[General Info]
Version=1
LastClass=CZjw002View
LastTemplate=CDialog
NewFileInclude1=#include "stdafx.h"
NewFileInclude2=#include "Zjw002.h"
LastPage=0

ClassCount=6
Class1=CZjw002App
Class2=CZjw002Doc
Class3=CZjw002View
Class4=CMainFrame

ResourceCount=3
Resource1=IDD_ABOUTBOX
Class5=CAboutDlg
Resource2=IDR_MAINFRAME
Class6=CHistogramDlg
Resource3=IDD_DIALOG1

[CLS:CZjw002App]
Type=0
HeaderFile=Zjw002.h
ImplementationFile=Zjw002.cpp
Filter=N

[CLS:CZjw002Doc]
Type=0
HeaderFile=Zjw002Doc.h
ImplementationFile=Zjw002Doc.cpp
Filter=N
BaseClass=CDocument
VirtualFilter=DC
LastObject=CZjw002Doc

[CLS:CZjw002View]
Type=0
HeaderFile=Zjw002View.h
ImplementationFile=Zjw002View.cpp
Filter=C
LastObject=ID_ILP_FILTER
BaseClass=CScrollView
VirtualFilter=VWC


[CLS:CMainFrame]
Type=0
HeaderFile=MainFrm.h
ImplementationFile=MainFrm.cpp
Filter=T
LastObject=ID_LINETRANS




[CLS:CAboutDlg]
Type=0
HeaderFile=Zjw002.cpp
ImplementationFile=Zjw002.cpp
Filter=D
LastObject=CAboutDlg

[DLG:IDD_ABOUTBOX]
Type=1
Class=CAboutDlg
ControlCount=4
Control1=IDC_STATIC,static,1342177283
Control2=IDC_STATIC,static,1342308480
Control3=IDC_STATIC,static,1342308352
Control4=IDOK,button,1342373889

[MNU:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_FILE_SAVE_AS
Command5=ID_FILE_PRINT
Command6=ID_FILE_PRINT_PREVIEW
Command7=ID_FILE_PRINT_SETUP
Command8=ID_FILE_MRU_FILE1
Command9=ID_APP_EXIT
Command10=ID_EDIT_UNDO
Command11=ID_EDIT_CUT
Command12=ID_EDIT_COPY
Command13=ID_EDIT_PASTE
Command14=ID_VIEW_TOOLBAR
Command15=ID_VIEW_STATUS_BAR
Command16=ID_APP_ABOUT
CommandCount=16

[ACL:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_NEW
Command2=ID_FILE_OPEN
Command3=ID_FILE_SAVE
Command4=ID_FILE_PRINT
Command5=ID_EDIT_UNDO
Command6=ID_EDIT_CUT
Command7=ID_EDIT_COPY
Command8=ID_EDIT_PASTE
Command9=ID_EDIT_UNDO
Command10=ID_EDIT_CUT
Command11=ID_EDIT_COPY
Command12=ID_EDIT_PASTE
Command13=ID_NEXT_PANE
Command14=ID_PREV_PANE
CommandCount=14

[TB:IDR_MAINFRAME]
Type=1
Class=CMainFrame
Command1=ID_FILE_OPEN
Command2=ID_APP_ABOUT
Command3=ID_GRAY
Command4=ID_LINETRANS
Command5=ID_Histogram
Command6=ID_LINETRANS
Command7=ID_EQUALIZE
Command8=ID_FT
Command9=ID_IFT
Command10=ID_FFT
Command11=ID_IFFT
Command12=ID_MED_FILETER
Command13=ID_GRAD_SHARP
Command14=ID_AVG_FILTER
Command15=ID_RAPLAS_SHARP
Command16=ID_FFT_FILTER
Command17=ID_ILP_FILTER
Command18=ID_GLP_FILTER
CommandCount=18

[DLG:IDD_DIALOG1]
Type=1
Class=CHistogramDlg
ControlCount=2
Control1=IDOK,button,1342242817
Control2=IDCANCEL,button,1342242816

[CLS:CHistogramDlg]
Type=0
HeaderFile=HistogramDlg.h
ImplementationFile=HistogramDlg.cpp
BaseClass=CDialog
Filter=D
VirtualFilter=dWC
LastObject=CHistogramDlg

