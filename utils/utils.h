#ifndef __UTILS_H__
#define __UTILS_H__


#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <math.h>
#include <stdexcept>
#include <string>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>



#define IN_ const
#define OUT_ 
#define not false ==
#define and &&
#define or ||


// a signed integer with 50 content bits (excluding the sign bit);
// ranged in (at least) [-10^15 , 10^15]; 
// has no precision loss when converted into a double;
// takes 8 bytes in physical memory
typedef long long int50;  



template<typename T>
T& vec_max(T* x, int50 n)
{	
	int50 i_max = 0;
	for(int50 i=0; i<n; i++)
	{
		i_max = (x[i] > x[i_max]) ? i : i_max;
	}
	return x[i_max];
}

template<typename T>
T& vec_min(T* x, int50 n)
{
	int50 i_min = 0;
	for(int50 i=0; i<n; i++)
	{
		i_min = (x[i] < x[i_min]) ? i : i_min;
	}
	return x[i_min];
}

template<typename T>
int50 vec_argmax(const T* x, int50 n)
{
	int50 i_max = 0;
	for(int50 i=0; i<n; i++)
	{
		i_max = (x[i] > x[i_max]) ? i : i_max;
	}
	return i_max;
}

template<typename T>
int50 vec_argmin(const T* x, int50 n)
{
	int50 i_min = 0;
	for(int50 i=0; i<n; i++)
	{
		i_min = (x[i] < x[i_min]) ? i : i_min;
	}
	return i_min;
}

double vec_sum(const double* x, int50 n)
{
	double sum = 0.0;
	for(int50 i=0; i<n; i++)
	{
		sum += x[i];
	}
	return sum;
}

double vec_avg(const double* x, int50 n)
{
	return vec_sum(x,n) / n;
}

double vec_norm_Lmax(const double* x, int50 n)
{
    double x_max = abs(x[0]);
	for(int50 i=1; i<n; i++)
	{
        double x_i = abs(x[i]);
        x_max = (x_i > x_max) ? x_i : x_max;
	}
	return x_max;    
}

double vec_norm_L0(const double* x, int50 n, double threshold =0)
{
    double cnt = 0;
	for(int50 i=1; i<n; i++)
	{
        cnt += (abs(x[i]) <= threshold) ? 0 : 1;
	}
	return cnt;    
}

double vec_norm_L1(const double* x, int50 n)
{
	double norm_l1 = 0.0;
	for(int50 i=0; i<n; i++)
	{
		norm_l1 += abs(x[i]);
	}
	return norm_l1;
}

double vec_avg_L1(const double* x, int50 n)
{
	return vec_norm_L1(x,n) / n;
}

double vec_dist_L1(const double* x, const double* y, int50 n)
{
	double dist_l1 = 0.0;
	for(int50 i=0; i<n; i++)
	{
		dist_l1 += abs(x[i]-y[i]);
	}
	return dist_l1;
}



int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


bool ExistFile(const char* filename)
{
	FILE* fp = fopen(filename, "r");
	if(fp == NULL) return false;
	fclose(fp);
	return true;
}

bool ClearFile(const char* filename)
{
	FILE* fp = fopen(filename, "w");
	if(fp == NULL) return false;
	fclose(fp);
	return true;	
}

bool RemoveFile(const char* filename)
{
	return (std::remove(filename) == 0);
}

// only support text files for now
class FileSynchronizer
{
public:
    std::string filename;
    FILE* fp;

public:
    FileSynchronizer(char* filename_str)
        : filename(filename_str), fp(NULL)
    {
        fp = fopen(filename.c_str(), "r");   
        if(fp == NULL) printf("[FileSync Error] cannot open %s to sync.\n", filename.c_str());
    }

    virtual ~FileSynchronizer()
    {
        if(fp != NULL) fclose(fp);
    }

    bool Sync(FILE* fp_dest =stdout)
    {
        if(fp_dest == NULL)
        {
            printf("[FileSync Error] destiation file is NULL.\n");
            return false;
        }

        char buf[5000];
        
        while(true)
        {
            int size = fread(buf, 1, 4096, fp);
            if(size == 0) break;

            buf[size] = 0;
            fwrite(buf, 1, size, fp_dest);
        }

        fflush(fp_dest);
        return true;
    }
};



// consume a constant amount of memory
// do not run construction and deconstruction functions
// support random access by index
template<typename _T>
class Queue
{
protected:
	int50 s;
	_T* item;
	void* mem;
	int50 h;
	int50 t;
	int50 CAPACITY;
	int50 MAX_SIZE;
	

public:
	Queue(int50 max_size) 
		: CAPACITY(max_size+1), MAX_SIZE(max_size), item(NULL), mem(NULL), s(0), h(0), t(0)
	{
		allocate(max_size);
	}

	virtual ~Queue() 
	{
		if(mem != NULL)
		{
			delete[] mem;
			mem = NULL;
			item = NULL;
		}
	}

public:
	int50 size() const {return s;}
	bool empty() const {return (s==0);} 
	bool full() const {return (s>=MAX_SIZE);} 

	void clear() {s=h=t=0;}
	void allocate(int50 max_size) 
	{
		if(mem != NULL)
		{
			delete[] mem;
			mem = NULL;
			item = NULL;
		}

		CAPACITY = max_size+1;
		MAX_SIZE = max_size;
		//printf("mem size=%lld\n", CAPACITY*sizeof(_T));
		mem = new char[CAPACITY*sizeof(_T)];	
		item = (_T*)mem;

		clear();
	}

	bool pop_front()
	{
		if(s < 1) 
		{
			throw std::invalid_argument( "[Queue warning] pop queue when it's empty" );
			return false;
		}
		assert(h!=t);

		h = (h+1)%CAPACITY;
		s --;
		return true;
	}

	bool push_back(const _T& new_item, bool auto_pop =false)
	{
		if(auto_pop == true) 
		{
			if(s == MAX_SIZE) 
				pop_front();
		}

		if(s >= MAX_SIZE) 
		{
			throw std::invalid_argument( "[Queue warning] push queue when it's full" );
			return false;
		}

		memcpy(item+t, &new_item, sizeof(_T));
		t = (t+1)%CAPACITY;
		s ++;
		return true;
	}

	_T& front() const {return item[h];}
	
	_T& back() const {return item[(t+CAPACITY-1)%CAPACITY];}

	_T& operator[](int50 i) const 
	{
		if(i>=s) 
		{
			printf("[Queue error] index out of scope: i=%lld, size=%lld\n", i, s);
			throw std::invalid_argument( "[Queue error] index out of scope" );
		}

		return item[(h+i)%CAPACITY];
	}



	//class QueueItem
	//{
	//public:
	//	int50 x[3];
	//	int50 y;

	//	QueueItem() { assert(0); }
	//	QueueItem(int50 yy) : y(yy) {}
	//};

	//static void unittest()
	//{
	//	Queue<QueueItem> queue(5);
	//	queue.pop_front();
	//	queue.push_back(QueueItem(1));
	//	queue.pop_front();
	//	queue.push_back(QueueItem(1));
	//	queue.push_back(QueueItem(2));
	//	queue.push_back(QueueItem(3));
	//	queue.push_back(QueueItem(4));
	//	queue.push_back(QueueItem(5));
	//	queue.push_back(QueueItem(6));
	//	printf("front=%lld, end=%lld, size=%lld\n", queue.front().y, queue.back().y, queue.size());
	//	for(int50 i=0; i<queue.size(); i++) printf("queue[%lld] = %lld\n", i, queue[i].y);
	//	while(queue.empty()==false) queue.pop_front();
	//	queue.push_back(QueueItem(6));
	//}
};



#endif