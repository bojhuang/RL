#pragma once

#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "EasyBMP.h"






bool Mat2BMP(unsigned char* x, int row, int column, char* filename)
{
	BMP img;
	img.SetSize(column,row);
	img.SetBitDepth(8);

	int id = 0;
	for(int i=0;i<row;i++)
	for(int j=0; j<column; j++)
	{
		img(j,i)->Red = img(j,i)->Blue = img(j,i)->Green = x[id++];
	}
	
	CreateGrayscaleColorTable(img);

	return img.WriteToFile(filename);
}

bool BMP2Mat(unsigned char* x, int row, int column, char* filename)
{
	BMP img;
	img.SetSize(column,row);
	img.SetBitDepth(8);

	if(img.ReadFromFile(filename) == false) return false;

	int id = 0;
	for(int i=0;i<row;i++)
	for(int j=0; j<column; j++)
	{
		x[id++] = img(j,i)->Red;// = img(j,i)->Blue = img(j,i)->Green = x[id++];
	}
	
	//CreateGrayscaleColorTable(img);

	return true;
}


int MatBMPTest()
{
	unsigned char d[100][100];
	for(int r=0; r<100; r++)
	{
		for(int c=0; c<100; c++)
		{
			d[r][c] = r;
		}
	}
	Mat2BMP((unsigned char*)d, 100, 100, "test.bmp");

	unsigned char e[100][100];
	BMP2Mat((unsigned char*)e, 100, 100, "test.bmp");

	for(int r=0; r<100; r++)
	{
		for(int c=0; c<100; c++)
		{
			if(d[r][c] != e[r][c]) printf("d(%d,%d)=%d e(%d,%d)=%d\n", r,c,d[r][c],r,c,e[r][c]);
		}
	}

	return 0;
}




