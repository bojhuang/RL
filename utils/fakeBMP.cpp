#include "EasyBMP.h"

bool BMP::SetBitDepth(int){}
bool BMP::WriteToFile(char const*){}
bool BMP::ReadFromFile(char const*){}
bool BMP::SetSize(int, int){}
BMP::BMP(){}
BMP::~BMP(){}
RGBApixel* BMP::operator()(int, int){ return NULL;}

bool CreateGrayscaleColorTable(BMP&){}