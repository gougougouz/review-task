#include "stdafx.h"

BITMAPINFO* lpBitsInfo = NULL;
BITMAPINFOHEADER bi;  // bi��һ��ָ����Ϣͷ��ָ��
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

// 24λ���ת�Ҷ�
void Gray24()
{
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // ÿһ����ռ���ֽ���
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // ָ��ͼ��ʵ�����ݵ�ָ��

	int LineBytes_gray = (w * 8 + 31) / 32 * 4;
	BITMAPINFO * lpBitsInfo_gray = (BITMAPINFO*)malloc(40 + 1024 + LineBytes_gray * h);

	memcpy(lpBitsInfo_gray,lpBitsInfo,40);
	lpBitsInfo_gray->bmiHeader.biBitCount = 8;
	lpBitsInfo_gray->bmiHeader.biClrUsed = 256;

	int i, j;
	// ����ɫ�帳ֵ
	for (i = 0; i < 256; i++) {
		lpBitsInfo_gray->bmiColors[i].rgbRed = i;
		lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
		lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
		lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
	}
	
	//��ȡ�Ҷ�ͼ�����ݵ�ָ��
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


// 2ֵת�Ҷ�
void Gray2()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w + 7) / 8;  //8 ������ 1 �ֽڼ���

    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // ָ��ͼ��ʵ�����ݵ�ָ��

    // Ϊ�Ҷ�ͼ������ڴ�
    int LineBytes_gray = (w + 3) / 4 * 4;  // ÿ������ռ 1 �ֽڣ�ȷ��ÿ���ֽ����� 4 �ı���
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(40 + 1024 + LineBytes_gray * h);

    // ����ͷ����Ϣ���޸�Ϊ�Ҷ�ͼ��
    memcpy(lpBitsInfo_gray, lpBitsInfo, 40);
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // ����Ϊ 8 λ�Ҷ�ͼ
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // ʹ�� 256 ɫ��ɫ��

    int i, j;
	// ����ɫ�帳ֵ
	for (i = 0; i < 256; i++) {
		lpBitsInfo_gray->bmiColors[i].rgbRed = i;
		lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
		lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
		lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
	}
	//��ȡ�Ҷ�ͼ�����ݵ�ָ��
    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];

    // ����ÿ�����أ���ȡλ������Ҷ�ֵ
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // ���㵱ǰ�ֽ���ͼ�������е�λ��
            int byteIndex = LineBytes * (h - 1 - i) + j / 8;
            int bitIndex = 7 - (j % 8);  // �������ҵ�˳��ȡÿ�����ص�λ

            // ��ȡ��ǰ����ֵ��0 �� 1��
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 1;

            // �� 0 ӳ��Ϊ�Ҷ�ֵ 1��1 ӳ��Ϊ�Ҷ�ֵ 255
            BYTE grayValue = (pixelValue == 0) ? 0 : 255;

            // ���Ҷ�ֵд��Ҷ�ͼ������
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = grayValue;
        }
    }

    free(lpBitsInfo);  // �ͷ�ԭʼͼ���ڴ�
    lpBitsInfo = lpBitsInfo_gray;  // ����Ϊ�Ҷ�ͼ��
}

// 16ɫת�Ҷ�
void Gray16()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // ÿ���ֽ���
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // ָ��ͼ��ʵ�����ݵ�ָ��

    // Ϊ�Ҷ�ͼ������ڴ�
    int LineBytes_gray = (w + 3) / 4 * 4;  // ÿ������ռ1�ֽڣ�ÿ�ж��뵽4�ı���
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);

    // ����ͷ����Ϣ���޸�Ϊ�Ҷ�ͼ��
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // ����Ϊ8λ�Ҷ�ͼ
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // ʹ��256ɫ��ɫ��

	int i, j;

    // ��ʼ���Ҷȵ�ɫ�壨ÿ���Ҷ�ֵ�� 0 �� 255��
    for (i = 0; i < 256; i++) {
        lpBitsInfo_gray->bmiColors[i].rgbRed = i;
        lpBitsInfo_gray->bmiColors[i].rgbGreen = i;
        lpBitsInfo_gray->bmiColors[i].rgbBlue = i;
        lpBitsInfo_gray->bmiColors[i].rgbReserved = 0;
    }

    BYTE* lpBits_gray = (BYTE*)&lpBitsInfo_gray->bmiColors[256];

    // ����ÿ������
    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            // ��ȡ16ɫͼ�������������2����1�ֽ�
            int byteIndex = LineBytes * (h - 1 - i) + j / 2;
            int bitIndex = (j % 2) ? 0 : 4;  // 4λ��ʾһ�����أ��� 2 ������һ���ֽ�

			// ��ȡ��ǰ���ص� 4 λֵ,һ��BYTE����ռһ���ֽ�
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F; 
            // ��ȡ��ɫ���е���ɫ
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];

            BYTE grayValue = (color.rgbRed + color.rgbGreen + color.rgbBlue) / 3;

            // ���Ҷ�ֵд��Ҷ�ͼ������
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = grayValue;
        }
    }

    free(lpBitsInfo);  // �ͷ�ԭʼͼ���ڴ�
    lpBitsInfo = lpBitsInfo_gray;  // ����Ϊ�Ҷ�ͼ��
}


// 256ɫת�Ҷ�
void Gray256()
{
    int w = lpBitsInfo->bmiHeader.biWidth;
    int h = lpBitsInfo->bmiHeader.biHeight;
    int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;  // ÿ���ֽ���
    BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed]; // ָ��ͼ��ʵ�����ݵ�ָ��

    // Ϊ�Ҷ�ͼ������ڴ�
    int LineBytes_gray = (w + 3) / 4 * 4;  // 8λ�Ҷ�ͼ��ÿ������ռ1�ֽ�
    BITMAPINFO* lpBitsInfo_gray = (BITMAPINFO*)malloc(sizeof(BITMAPINFOHEADER) + 256 * sizeof(RGBQUAD) + LineBytes_gray * h);
    
    // ����ͷ����Ϣ���޸�Ϊ�Ҷ�ͼ��
    memcpy(lpBitsInfo_gray, lpBitsInfo, sizeof(BITMAPINFOHEADER));
    lpBitsInfo_gray->bmiHeader.biBitCount = 8;   // ����Ϊ8λ�Ҷ�ͼ
    lpBitsInfo_gray->bmiHeader.biClrUsed = 256;  // ʹ��256ɫ��ɫ��

    // ��ʼ���Ҷȵ�ɫ�壨ÿ���Ҷ�ֵ�� 0 �� 255��
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
            // ȡ��ͼ���е�ÿ�����ص���ɫ����
            BYTE pixelIndex = lpBits[LineBytes * (h - 1 - i) + j];
            
            // ��ȡ��ɫ���е���ɫ
            RGBQUAD color = lpBitsInfo->bmiColors[pixelIndex];
            
            BYTE avg = (color.rgbRed + color.rgbGreen + color.rgbBlue) / 3;
            
            // ���Ҷ�ֵ����Ҷ�ͼ������
            lpBits_gray[LineBytes_gray * (h - 1 - i) + j] = avg;
        }
    }

    free(lpBitsInfo);  // �ͷ�ԭʼͼ����ڴ�
    lpBitsInfo = lpBitsInfo_gray;  // ����Ϊ�Ҷ�ͼ��
}


void Gray() {
	switch(bi.biBitCount)
	{
	case 1:  //��ֵͼ��
		Gray2();
		break;
	case 4:  //16ɫͼ��
		Gray16();
		break;
	case 8: //256ɫͼ��
		Gray256();
		break;
	case 24: //24λ���
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
			sprintf(str, "�Ҷ�ֵ: %d", *pixel);
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
			strcpy(str,"������");
		else
			strcpy(str,"ǰ����");

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
			// �Ҷ�ͼ�������������Ҷ�ֵ
			pixel = lpBits + LineBytes * (h - 1 - i) + j; 
			H[*pixel]++;
		}
	}
}

DWORD R[256];
DWORD G[256];
DWORD B[256];

// 24λ���ͼ���rgbͨ����ֱ��ͼ
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
			// ÿһ������ռ�����ֽ�
			pb = lpBits + LineBytes * (h - 1 - i) + j * 3;
			pg = pb + 1;
			pr = pg + 1;
			B[*pb] ++;
			G[*pg] ++;
			R[*pr] ++;
		}
	}
}

// 16λ���ͼ���rgbͨ����ֱ��ͼ
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
			// 4λ��ʾһ�����أ�2 ������һ���ֽڣ�jΪż��ǰ��λ��jΪ��������λ������λ��������
            int bitIndex = (j % 2) ? 0 : 4;  

			// ��ȡ��ǰ���ص� 4 λֵ,һ��BYTE����ռһ���ֽ�
            BYTE pixelValue = (lpBits[byteIndex] >> bitIndex) & 0x0F;  

			// ��ȡ��ɫ���е���ɫ
            RGBQUAD color = lpBitsInfo->bmiColors[pixelValue];

            R[color.rgbRed] += 1;
			G[color.rgbGreen] += 1;
			B[color.rgbBlue] +=1 ;
		}
	}
}

// 256λ���ͼ���rgbͨ����ֱ��ͼ
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
			// ȡ��ͼ���е�ÿ�����ص���ɫ����
            BYTE pixelIndex = lpBits[LineBytes * (h - 1 - i) + j];
            
            // ��ȡ��ɫ���е���ɫ
            RGBQUAD color = lpBitsInfo->bmiColors[pixelIndex];
		
            R[color.rgbRed] += 1;
			G[color.rgbGreen] += 1;
			B[color.rgbBlue] +=1 ;

		}
	}
}

// ����ͼƬ���͵�����Ӧ��ֱ��ͼ���㺯��
void Histogram()
{
	if(IsGray()){
		HistogramGray();
		return;
	}
    switch (bi.biBitCount) {
        case 4:  // 16ɫͼ��
            Histogram16();
            break;
        case 8:  // 256ɫͼ��
            Histogram256();
            break;
        case 24: // 24λ���ͼ��
            Histogram24();
            break;
    }
}

// ȫ�ֱ�������������ӱ༭���õĲ���
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
	
	// ��ԻҶ�ͼ��
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
            pixel = lpBits + LineBytes * (h - 1 - i) + j;
            temp = a * (*pixel) + b;
			
			// �Ե�ǰ���ؽ��е�ֵ���и���
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

// �Ի�ɫͼ�������⻯
void EqualizeGray() {
	int w = lpBitsInfo->bmiHeader.biWidth;
	int h = lpBitsInfo->bmiHeader.biHeight;
	int LineBytes = (w * lpBitsInfo->bmiHeader.biBitCount + 31) / 32 * 4;
	BYTE* lpBits = (BYTE*)&lpBitsInfo->bmiColors[lpBitsInfo->bmiHeader.biClrUsed];

	int i, j; 
	BYTE* pixel;
	DWORD temp = 0;

	int Map[256];
	
	HistogramGray(); // ����ֱ��ͼ��������H

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

// ��24λ���ͼ�������⻯
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

	Histogram24();  //����ֱ��ͼ��������R��G��B
	
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

	Histogram16();  //����ֱ��ͼ��������R��G��B
	
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

	Histogram256();  //����ֱ��ͼ��������R��G��B
	
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
        case 4:  // 16ɫͼ��
            Equalize16();
            break;
        case 8:  // 256ɫͼ��
            Equalize256();
            break;
        case 24: // 24λ���ͼ��
            Equalize24();
            break;
    }
}