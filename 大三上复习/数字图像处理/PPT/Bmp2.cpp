#include "stdafx.h"

BITMAPINFO* lpBitsInfo = NULL;
BITMAPINFOHEADER bi;  // bi是一个指向信息头的指针
BOOL LoadBmpFile(char* BmpFileName)
{
	FILE *fp;
	if (NULL == (fp = fopen(BmpFileName,"rb")))
		return FALSE;

	BITMAPFILEHEADER bf;
	//BITMAPINFOHEADER bi;

	fread(&bf, 14, 1, fp); 
	fread(&bi, 40, 1, fp);
	
	DWORD NumColors;
	if (bi.biClrUsed != 0) 
		NumColors = bi.biClrUsed;
	else
	{
		switch(bi.biBitCount)
		{
		case 1:
			NumColors = 2;
			break;
		case 4:
			NumColors = 16;
			break;
		case 8:
			NumColors = 256;
			break;
		case 24:
			NumColors = 0;
			break;
		}
	}
	DWORD PalSize = NumColors * 4;
	DWORD ImgSize = (bi.biWidth * bi.biBitCount + 31) / 32 * 4 * bi.biHeight;
	DWORD Size = 40 + PalSize + ImgSize;

	if ( NULL == (lpBitsInfo = (BITMAPINFO*)malloc(Size)))
		return FALSE;

	fseek(fp, 14, SEEK_SET);
	fread((char*)lpBitsInfo,Size,1,fp);

	lpBitsInfo->bmiHeader.biClrUsed = NumColors;

	return TRUE;
}

// 24位真彩转灰度
void Gray24()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // 每一行所占的字节数
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // 指向图像实际数据的指针

	int LineBytes_gray = (w * 8 + 31) / 32 * 4;
	BITMAPINFO * lpBitsInfo_gray = (BITMAPINFO*)malloc(40 + 1024 + LineBytes_gray * h);

	memcpy(lpBitsInfo_gray,lpBitsInfo,40);
	lpBitsInfo_gray->bmiHeader.biBitCount = 8;
	lpBitsInfo_gray->bmiHeader.biClrUsed = 256;

	int i, j;
	// 给调色板赋值
	for (i = 0; i < 256; i++) {
		lpBitsInfo_gray->bmiColors[i].rgbRed = i;
		lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
		lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
		lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
	}
	
	//获取灰度图像数据的指针
	BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];
	BYTE *R, *G, *B,avg, *pixel;
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			B = lpBits + LineBytes * (h - 1 - i) + j * 3;
			G = B + 1;
			R = G + 1;
			avg = (*R + *G + *B) / 3;
			pixel = lpBits_gray + LineBytes_gray * (h - 1 - i) + j;
			*pixel = avg;
		}
	}
	free(lpBitsInfo);
	lpBitsInfo = lpBitsInfo_gray;
}


// 2值转灰度
void Gray2()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w + 7) / 8;  //8 个像素 1 字节计算

    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // 指向图像实际数据的指针

    // 为灰度图像分配内存
    int LineBytes_gray = (w + 3) / 4 * 4;  // 每个像素占 1 字节，确保每行字节数是 4 的倍数
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(40 + 1024 + LineBytes_gray * h);

    // 复制头部信息并修改为灰度图像
    memcpy(lpBitsInfo_gray, lpBitsInfo, 40);
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // 设置为 8 位灰度图
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // 使用 256 色调色板

    int i, j;
	// 给调色板赋值
	for (i = 0; i < 256; i++) {
		lpBitsInfo_gray->bmiColors[i].rgbRed = i;
		lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
		lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
		lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
	}
	//获取灰度图像数据的指针
    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];

    // 遍历每个像素，提取位并计算灰度值
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // 计算当前字节在图像数据中的位置
            int byteIndex = LineBytes * (h - 1 - i) + j / 8;
            int bitIndex = 7 - (j % 8);  // 按从左到右的顺序取每个像素的位

            // 获取当前像素值（0 或 1）
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 1;

            // 将 0 映射为灰度值 1，1 映射为灰度值 255
            BYTE grayValue = (pixelValue == 0) ? 0 : 255;

            // 将灰度值写入灰度图像数据
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = grayValue;
        }
    }

    free(lpBitsInfo);  // 释放原始图像内存
    lpBitsInfo = lpBitsInfo_gray;  // 更新为灰度图像
}

// 16色转灰度
void Gray16()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // 每行字节数
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // 指向图像实际数据的指针

    // 为灰度图像分配内存
    int LineBytes_gray = (w + 3) / 4 * 4;  // 每个像素占1字节，每行对齐到4的倍数
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);

    // 复制头部信息并修改为灰度图像
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // 设置为8位灰度图
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // 使用256色调色板

	int i, j;

    // 初始化灰度调色板（每个灰度值从 0 到 255）
    for (i = 0; i < 256; i++) {
        lpBitsInfo_gray->bmiColors[i].rgbRed = i;
        lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
        lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
        lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
    }

    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];

    // 遍历每个像素
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // 获取16色图像的像素索引，2像素1字节
            int byteIndex = LineBytes * (h - 1 - i) + j / 2;
            int bitIndex = (j % 2) ? 0 : 4;  // 4位表示一个像素，按 2 个像素一个字节

			// 获取当前像素的 4 位值,一个BYTE类型占一个字节
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F; 
            // 获取调色板中的颜色
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];

            BYTE grayValue = (color.rgbRed + color.rgbGreen + color.rgbBlue) / 3;

            // 将灰度值写入灰度图像数据
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = grayValue;
        }
    }

    free(lpBitsInfo);  // 释放原始图像内存
    lpBitsInfo = lpBitsInfo_gray;  // 更新为灰度图像
}


// 256色转灰度
void Gray256()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // 每行字节数
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // 指向图像实际数据的指针

    // 为灰度图像分配内存
    int LineBytes_gray = (w + 3) / 4 * 4;  // 8位灰度图像每个像素占1字节
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);
    
    // 复制头部信息并修改为灰度图像
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // 设置为8位灰度图
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // 使用256色调色板

    // 初始化灰度调色板（每个灰度值从 0 到 255）
	int i, j;
    for (i = 0; i < 256; i++) {
        lpBitsInfo_gray->bmiColors[i].rgbRed = i;
        lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
        lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
        lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
    }

    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // 取出图像中的每个像素的颜色索引
            BYTE pixelIndex = lpBits[LineBytes * (h - 1 - i) + j];
            
            // 获取调色板中的颜色
            RGBQUAD color = lpBitsInfo->bmiColors[pixelIndex];
            
            BYTE avg = (color.rgbRed + color.rgbGreen + color.rgbBlue) / 3;
            
            // 将灰度值存入灰度图像数据
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = avg;
        }
    }

    free(lpBitsInfo);  // 释放原始图像的内存
    lpBitsInfo = lpBitsInfo_gray;  // 更新为灰度图像
}


void Gray() {
	switch(bi.biBitCount)
	{
	case 1:  //二值图像
		Gray2();
		break;
	case 4:  //16色图像
		Gray16();
		break;
	case 8: //256色图像
		Gray256();
		break;
	case 24: //24位真彩
		Gray24();
		break;
	}
}

BOOL IsGray(){
	int r, g, b;
	if (8 == lpBitsInfo->bmiHeader.biBitCount)
	{
		r = lpBitsInfo->bmiColors[128].rgbRed;
		g = lpBitsInfo->bmiColors[128].rgbGreen;
		b = lpBitsInfo->bmiColors[128].rgbBlue;

		if (r == b && r == g)
			return TRUE;
	}
	return FALSE;
}

void pixel(int i, int j, char* str){
	if (NULL == lpBitsInfo)
		return;

	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	if (i >= h || j >= w)
		return;

	BYTE* pixel, bv;
	int r, g, b;
	switch(lpBitsInfo->bmiHeader.biBitCount)
	{
	case 8:
		pixel = lpBits + LineBytes * (h - 1 - i) + j;
		if (IsGray())
			sprintf(str, "灰度值: %d", *pixel);
		else {
			r = lpBitsInfo->bmiColors[*pixel].rgbRed;
			g = lpBitsInfo->bmiColors[*pixel].rgbGreen;
			b = lpBitsInfo->bmiColors[*pixel].rgbBlue;
			sprintf(str, "RGB(%d, %d, %d)",r, g, b);
		}
		break;
	case 24:
		break;
	case 4:
		break;
	case 1:
		bv = *(lpBits + LineBytes * (h - 1 - i) + j/8) & (1 << (j % 8));
		if (0 == bv)
			strcpy(str,"背景点");
		else
			strcpy(str,"前景点");

		break;
	}
}

DWORD H[256];

void HistogramGray()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;
	BYTE* pixel;

	for(i = 0; i < 256; i++)
		H[i] = 0;

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			// 灰度图像的索引就是其灰度值
			pixel = lpBits + LineBytes * (h - 1 - i) + j; 
			H[*pixel]++;
		}
	}
}

DWORD R[256];
DWORD G[256];
DWORD B[256];

// 24位真彩图像的rgb通道的直方图
void Histogram24()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;
	BYTE* pb, *pr, *pg;

	for(i = 0; i < 256; i++) {
		R[i] = 0;
		G[i] = 0;
		B[i] = 0;
	}

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			// 每一个像素占三个字节
			pb = lpBits + LineBytes * (h - 1 - i) + j * 3;
			pg = pb + 1;
			pr = pg + 1;
			B[*pb] ++;
			G[*pg] ++;
			R[*pr] ++;
		}
	}
}

// 16位真彩图像的rgb通道的直方图
void Histogram16()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;

	for(i = 0; i < 256; i++) {
		R[i] = 0;
		G[i] = 0;
		B[i] = 0;
	}

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			int byteIndex = LineBytes * (h - 1 - i) + j / 2;
			// 4位表示一个像素，2 个像素一个字节，j为偶数前四位，j为奇数后四位，后四位不用右移
            int bitIndex = (j % 2) ? 0 : 4;  

			// 获取当前像素的 4 位值,一个BYTE类型占一个字节
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F;  

			// 获取调色板中的颜色
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];

            R[color.rgbRed] += 1;
			G[color.rgbGreen] += 1;
			B[color.rgbBlue] +=1 ;
		}
	}
}

// 256位真彩图像的rgb通道的直方图
void Histogram256()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;

	for(i = 0; i < 256; i++) {
		R[i] = 0;
		G[i] = 0;
		B[i] = 0;
	}

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			// 取出图像中的每个像素的颜色索引
            BYTE pixelIndex = lpBits[LineBytes * (h - 1 - i) + j];
            
            // 获取调色板中的颜色
            RGBQUAD color = lpBitsInfo->bmiColors[pixelIndex];
		
            R[color.rgbRed] += 1;
			G[color.rgbGreen] += 1;
			B[color.rgbBlue] +=1 ;

		}
	}
}

// 根据图片类型调用相应的直方图计算函数
void Histogram()
{
	if(IsGray()){
		HistogramGray();
		return;
	}
    switch (bi.biBitCount) {
        case 4:  // 16色图像
            Histogram16();
            break;
        case 8:  // 256色图像
            Histogram256();
            break;
        case 24: // 24位真彩图像
            Histogram24();
            break;
    }
}

// 全局变量，用来保存从编辑框获得的参数
double k = 1;
double b = 1;

void LineTrans(double a, double b) {
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;
	BYTE *pixel;
	double temp;
	
	// 针对灰度图像
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
            pixel = lpBits + LineBytes * (h - 1 - i) + j;
            temp = a * (*pixel) + b;
			
			// 对当前像素进行的值进行更新
			if (temp > 255){
				*pixel = 255;
			}
			else if (temp < 0) {
				*pixel = 0;
			}
			else {
				*pixel = (BYTE) (temp + 0.5);
			}
		}
	}
}

// 对灰色图像做均衡化
void EqualizeGray() {
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j; 
	BYTE* pixel;
	DWORD temp = 0;

	int Map[256];
	
	HistogramGray(); // 调用直方图函数更新H

	for (i = 0; i < 256; i ++) {
		temp += H[i];
		Map[i] = 255 * temp / (w * h);
	}
	
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
            pixel = lpBits + LineBytes * (h - 1 - i) + j;
            *pixel = Map[*pixel];
		}
	}
}

// 对24位真彩图像做均衡化
void Equalize24() {
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;
	BYTE* pixel;
	DWORD tempr = 0;
	DWORD tempg = 0;
	DWORD tempb = 0;

	int Mapr[256], Mapg[256], Mapb[256];

	Histogram24();  //调用直方图函数更新R，G，B
	
	for (i = 0; i < 256; i ++) {
		tempr += R[i];
		tempg+= G[i];
		tempb+= B[i];

		Mapr[i] = 255 * tempr / (w * h);
		Mapg[i] = 255 * tempg / (w * h);
		Mapb[i] = 255 * tempb / (w * h);
	}
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
            pixel = lpBits + LineBytes * (h - 1 - i) + 3 * j;
            *pixel = Mapb[*pixel];
			*(pixel + 1) = Mapg[*(pixel+1)];
			*(pixel + 2) = Mapr[*(pixel+2)];
		}
	}
}

void Equalize16() {
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;

	DWORD tempr = 0;
	DWORD tempg = 0;
	DWORD tempb = 0;

	int Mapr[256], Mapg[256], Mapb[256];

	Histogram16();  //调用直方图函数更新R，G，B
	
	for (i = 0; i < 256; i ++) {
		tempr += R[i];
		tempg+= G[i];
		tempb+= B[i];

		Mapr[i] = 255 * tempr / (w * h);
		Mapg[i] = 255 * tempg / (w * h);
		Mapb[i] = 255 * tempb / (w * h);
	}

	RGBQUAD newPalette[16];

	for (i = 0; i < 16; i++) {
		 RGBQUAD color = lpBitsInfo->bmiColors[i];
		 color.rgbRed = Mapr[color.rgbRed];
		 color.rgbBlue = Mapb[color.rgbBlue];
		 color.rgbGreen = Mapg[color.rgbGreen];

		 newPalette[i] = color;
	}

	for (j = 0; j < 16; j++) {
		lpBitsInfo->bmiColors[j] = newPalette[j];
	}
}


void Equalize256() {
    int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;

	DWORD tempr = 0;
	DWORD tempg = 0;
	DWORD tempb = 0;

	int Mapr[256], Mapg[256], Mapb[256];

	Histogram256();  //调用直方图函数更新R，G，B
	
	for (i = 0; i < 256; i ++) {
		tempr += R[i];
		tempg+= G[i];
		tempb+= B[i];

		Mapr[i] = 255 * tempr / (w * h);
		Mapg[i] = 255 * tempg / (w * h);
		Mapb[i] = 255 * tempb / (w * h);
	}

	RGBQUAD newPalette[256];

	for (i = 0; i < 256; i++) {
		 RGBQUAD color = lpBitsInfo->bmiColors[i];
		 color.rgbRed = Mapr[color.rgbRed];
		 color.rgbBlue = Mapb[color.rgbBlue];
		 color.rgbGreen = Mapg[color.rgbGreen];

		 newPalette[i] = color;
	}

	for (j = 0; j < 256; j++) {
		lpBitsInfo->bmiColors[j] = newPalette[j];
	}
}

void Equalize() {
	if(IsGray()){
		EqualizeGray();
		return;
	}
    switch (bi.biBitCount) {
        case 4:  // 16色图像
            Equalize16();
            break;
        case 8:  // 256色图像
            Equalize256();
            break;
        case 24: // 24位真彩图像
            Equalize24();
            break;
    }
}