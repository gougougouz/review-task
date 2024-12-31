#include "stdafx.h"
#include "windows.h"
BITMAPINFO* lpBitsInfo = NULL;
BITMAPINFOHEADER bi;
BOOL LoadBmpFile(char* BmpFileName){
	FILE* fp;
	if(NULL == (fp = fopen(BmpFileName,"rb")))
		return FALSE;

	BITMAPFILEHEADER bf;


	fread(&bf, 14, 1, fp);
	fread(&bi, 40, 1, fp);

	DWORD NumColors;
	if (bi.biClrUsed != 0)
		NumColors = bi.biClrUsed;
	else{
		switch(bi.biBitCount){
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

	if(NULL == (lpBitsInfo = (BITMAPINFO*)malloc(Size)))
		return FALSE;

	fseek(fp, 14, SEEK_SET);
	fread((char*)lpBitsInfo, Size,1,fp);

	lpBitsInfo->bmiHeader.biClrUsed = NumColors;
    return TRUE;
}


void Gray2()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w + 7) / 8;  // 每行字节数，按 8 个像素 1 字节计算

    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // 指向图像实际数据的指针

    // 为灰度图像分配内存
    int LineBytes_gray = (w + 3) / 4 * 4;  // 每个像素占 1 字节，确保每行字节数是 4 的倍数
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);

    // 复制头部信息并修改为灰度图像
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // 设置为 8 位灰度图
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // 使用 256 色调色板

    // 初始化灰度调色板（黑白，灰度值 1 和 255）
    lpBitsInfo_gray->bmiColors[0].rgbRed = 1;
    lpBitsInfo_gray->bmiColors[0].rgbGreen = 1;
    lpBitsInfo_gray->bmiColors[0].rgbBlue = 1;
    lpBitsInfo_gray->bmiColors[0].rgbReserved = 0;

    lpBitsInfo_gray->bmiColors[255].rgbRed = 255;
    lpBitsInfo_gray->bmiColors[255].rgbGreen = 255;
    lpBitsInfo_gray->bmiColors[255].rgbBlue = 255;
    lpBitsInfo_gray->bmiColors[255].rgbReserved = 0;

    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];

	int i, j;

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



void Gray16()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // 每行字节数
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[16]; // 固定指向16色调色板之后的数据

    // 为灰度图像分配内存
    int LineBytes_gray = (w + 3) & ~3;  // 每个像素占1字节，每行对齐到4的倍数
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);

    // 复制头部信息并修改为灰度图像
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // 设置为8位灰度图
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // 使用256色调色板

    int i,j;

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
            int byteIndex = LineBytes * (h - 1 - i) + j / 2;
            int bitIndex = (j % 2) ? 0 : 4;

            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F;  // 获取当前像素的 4 位值

            // 获取调色板中的颜色
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];

            // 加权平均计算灰度值
            BYTE grayValue = (BYTE)((color.rgbRed + color.rgbGreen + color.rgbBlue)/3);

            // 将灰度值写入灰度图像数据
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = grayValue;
        }
    }

    free(lpBitsInfo);  // 释放原始图像内存
    lpBitsInfo = lpBitsInfo_gray;  // 更新为灰度图像
}


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
            
            // 获取调色板中的颜色（256色图像是从调色板获取颜色）
            RGBQUAD color = lpBitsInfo->bmiColors[pixelIndex];
            
            // 计算灰度值，这里使用简单的平均法
            BYTE avg = (color.rgbRed + color.rgbGreen + color.rgbBlue) / 3;
            
            // 将灰度值存入灰度图像数据
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = avg;
        }
    }

    free(lpBitsInfo);  // 释放原始图像的内存
    lpBitsInfo = lpBitsInfo_gray;  // 更新为灰度图像
}


void Gray24(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int LineBytes_gray = (w * 8 + 31)/32 * 4;
	BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(40 + 1024 + LineBytes_gray * h);

	memcpy(lpBitsInfo_gray, lpBitsInfo, 40);
	lpBitsInfo_gray->bmiHeader.biBitCount = 8;
	lpBitsInfo_gray->bmiHeader.biClrUsed = 256;

	int i, j;
	for(i = 0;i < 256;i ++){
		lpBitsInfo_gray->bmiColors[i].rgbRed = i;
		lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
		lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
		lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
	}

	BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];
 
	BYTE *R,*G,*B,avg,*pixel;
    lpBitsInfo->bmiHeader.biBitCount;

	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			B = lpBits + LineBytes * (h - 1 - i) + j * 3;
			G = B + 1;
			R = G + 1;
			avg = (*R + *G + *B)/3;
			pixel = lpBits_gray + LineBytes_gray * (h - 1 - i) + j;
			*pixel = avg;
		}
	}
	free(lpBitsInfo);
	lpBitsInfo = lpBitsInfo_gray;
}

void Gray(){
    switch(bi.biBitCount)
	{
	case 8:
		Gray256();
		break;
	case 24:
		Gray24();
		break;
	case 4:
		Gray16();
		break;
	case 1:
		Gray2();
		break;
	}
}

BOOL IsGray()
{
	int r,g,b;
	if (8 == lpBitsInfo->bmiHeader.biBitCount)
	{
		r = lpBitsInfo->bmiColors[150].rgbRed;
		g = lpBitsInfo->bmiColors[150].rgbGreen;
		b = lpBitsInfo->bmiColors[150].rgbBlue;
           
		if(r == b && r == g)
			return TRUE;
	}
	return FALSE;

}
void pixel(int i, int j, char* str)
{
	if(NULL == lpBitsInfo)
		return;

    int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	if (i >= h || j >=w)
		return;

	BYTE* pixel, bv;
	int r,g,b;
	int colorIdx = 0;

	switch(lpBitsInfo->bmiHeader.biBitCount)
	{
	case 8:

		pixel = lpBits + LineBytes * (h - 1 - i) + j;
		if (IsGray())
            sprintf(str,"灰度:%d", *pixel);
		else
		{

			r = lpBitsInfo->bmiColors[*pixel].rgbRed;
			g = lpBitsInfo->bmiColors[*pixel].rgbGreen;
			b = lpBitsInfo->bmiColors[*pixel].rgbBlue;
			sprintf(str,"RGB(%d, %d, %d)", r,g,b);
		}
		break;
	case 24:
		pixel = lpBits + LineBytes * (h - 1 - i) + j * 3;
		r = pixel[0];
		g = pixel[1];
		b = pixel[2];
		sprintf(str,"RGB(%d, %d, %d)", r,g,b);
		break;
	case 4:
		bv = *(lpBits + LineBytes * (h - 1 - i) + j / 2);
		colorIdx = (j % 2 == 0) ? (bv >> 4) : (bv & 0x0f);
		r = lpBitsInfo->bmiColors[colorIdx].rgbRed;
		g = lpBitsInfo->bmiColors[colorIdx].rgbGreen;
		b = lpBitsInfo->bmiColors[colorIdx].rgbBlue;
		sprintf(str,"RGB(%d, %d, %d)", r,g,b);
		break;
	case 1:
		bv = *(lpBits + LineBytes * (h - 1 - i) + j/8) & (1 << (7 - j % 8));
		if(0 == bv)
			strcpy(str,"背景点");
		else
			strcpy(str,"前景点");
		break;
	}
}

DWORD H[256];
void Histogram()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	BYTE* pixel;
	for(i = 0; i < 256; i++)
		H[i] = 0;

	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
			H[*pixel] ++;
		}
	}

}
DWORD HR[256], HG[256], HB[256];
void H256(){
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

    int i,j;
    for (i = 0; i < 256; i++) HR[i] = HG[i] = HB[i] = 0;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            BYTE* pixel = lpBits + LineBytes * (h - 1 - i) + j;
			RGBQUAD color = lpBitsInfo->bmiColors[* pixel];
            HR[color.rgbRed]++;
            HG[color.rgbGreen]++;
            HB[color.rgbBlue]++;
        }
    }
}

// 24位位图的直方图计算函数
void H24() {
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];
	BYTE *R,*G,*B;

	int i,j;
    for (i = 0; i < 256; i++) HR[i] = HG[i] = HB[i] = 0;
    for (i = 0; i < h; i++) {
        BYTE* line = lpBits + i * LineBytes;
        for (j = 0; j < w; j++) {
			B = lpBits + LineBytes * (h - 1 - i) + j * 3;
			G = B + 1;
			R = G + 1;
            HR[*R]++;
            HG[*G]++;
            HB[*B]++;
        }
    }
}

void H16()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j;

	for(i = 0; i < 256; i++) {
		HR[i] = 0;
		HG[i] = 0;
		HB[i] = 0;
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

            HR[color.rgbRed] += 1;
			HG[color.rgbGreen] += 1;
			HB[color.rgbBlue] +=1 ;
		}
	}
}


void His(){
	if(IsGray()){
		Histogram();
	}
	else{
		switch(bi.biBitCount){
			case 8:
				H256();
				break;
			case 24:
				H24();
				break;
			case 4:
				H16();
				break;
		}
	}
}

void LineTrans(float a,float b){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	BYTE* pixel;
	float temp;
    for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
			temp = a * (*pixel) + b;
			if(temp > 255)
				*pixel = 255;
			else if(temp < 0)
				*pixel = 0;
			else
				*pixel = (BYTE)(temp + 0.5);
		}
	}
}

void Equalize(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	BYTE* pixel;
	DWORD temp;

	BYTE Map[256];
	His();

	for(i = 0;i < 256; i++){
		temp = 0;
		for(j = 0;j <= i; j++){
			temp += H[j];
		}
		Map[i] = 255 * temp / (h * w);
	}

	for(i = 0;i < h; i ++){
		for(j = 0;j < w;j ++){
			pixel = lpBits + LineBytes * (h - 1 - i) + j;
			*pixel = Map[*pixel];
		}
	}
}

void E256(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;
	DWORD tempr,tempg,tempb;

	RGBQUAD* palette = lpBitsInfo->bmiColors;
	BYTE MapR[256],MapG[256],MapB[256];
	His();
    for(i = 0;i < 256; i++){
		tempr = 0;
		tempg = 0;
		tempb = 0;
		for(j = 0;j <= i; j++){
			tempr += HR[j];
			tempg += HG[j];
			tempb += HB[j];
		}
		MapR[i] = 255.0 * tempr / (h * w);
		MapG[i] = 255.0 * tempg / (h * w);
		MapB[i] = 255.0 * tempb / (h * w);
	}
    
	RGBQUAD newPalette[256];  
    for(i = 0; i < 256; i++){  

        newPalette[i].rgbRed = MapR[palette[i].rgbRed];  
        newPalette[i].rgbGreen = MapG[palette[i].rgbGreen];  
        newPalette[i].rgbBlue = MapB[palette[i].rgbBlue];  
    } 
    
    // 更新调色板  
    for(i = 0; i < 256; i++){  
        palette[i] = newPalette[i];  
    }  
}


void E24(){
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31)/32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i,j;

	DWORD tempr,tempg,tempb;
	BYTE *R,*G,*B;

	BYTE MapR[256],MapG[256],MapB[256];
	His();
    for(i = 0;i < 256; i++){
		tempr = 0;
		tempg = 0;
		tempb = 0;
		for(j = 0;j <= i; j++){
			tempr += HR[j];
			tempg += HG[j];
			tempb += HB[j];
		}
		MapR[i] = 255.0 * tempr / (h * w);
		MapG[i] = 255.0 * tempg / (h * w);
		MapB[i] = 255.0 * tempb / (h * w);
	}
     
    for (i = 0; i < h; i++) {
        BYTE* line = lpBits + i * LineBytes;
        for (j = 0; j < w; j++) {
			B = lpBits + LineBytes * (h - 1 - i) + j * 3;
			G = B + 1;
			R = G + 1;
		    *B = MapB[*B];
			*G = MapG[*G];
            *R = MapR[*R];
        }
    }

}

void E16(){
	int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[16];


	int i,j;
	DWORD tempr,tempg,tempb;

	RGBQUAD* palette = lpBitsInfo->bmiColors;
	BYTE MapR[256],MapG[256],MapB[256];
	His();
    for(i = 0;i < 256; i++){
		tempr = 0;
		tempg = 0;
		tempb = 0;
		for(j = 0;j <= i; j++){
			tempr += HR[j];
			tempg += HG[j];
			tempb += HB[j];
		}
		MapR[i] = 255.0 * tempr / (h * w);
		MapG[i] = 255.0 * tempg / (h * w);
		MapB[i] = 255.0 * tempb / (h * w);
	}
    
	RGBQUAD newPalette[16];  
    for(i = 0; i < 16; i++){  
        newPalette[i].rgbRed = MapR[palette[i].rgbRed];  
        newPalette[i].rgbGreen = MapG[palette[i].rgbGreen];  
        newPalette[i].rgbBlue = MapB[palette[i].rgbBlue];  
    }  
    
    // 更新调色板  
    for(i = 0; i < 16; i++){  
        palette[i] = newPalette[i];  
    }  

}


void Eql(){
	if(IsGray()){
		Equalize();
	}
	else{
		switch(bi.biBitCount){
		case 8:
			E256();
			break;
		case 24:
			E24();
			break;
		case 4:
			E16();
			break;
		}
	}
}