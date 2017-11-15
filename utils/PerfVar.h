#ifndef __PERFVAR_H__
#define __PERFVAR_H__


#include <stdio.h>
#include <math.h>

#include <assert.h>

#if defined(__linux__) || defined(__APPLE__)
#include <inttypes.h>
#include <chrono>
#include <thread>

#define __int64 long long

typedef std::chrono::high_resolution_clock::time_point LARGE_INTEGER;

inline void QueryPerformanceCounter(std::chrono::high_resolution_clock::time_point *timepoint)
{
    *timepoint = std::chrono::high_resolution_clock::now();
}

double toMicroSeconds(LARGE_INTEGER endTime, LARGE_INTEGER startTime)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
}

#else
#include <windows.h>
#include <process.h>

double toMicroSeconds(LARGE_INTEGER endTime, LARGE_INTEGER startTime, LARGE_INTEGER frequency)
{
    return (endTime.QuadPart - startTime.QuadPart)
                    / (double) (frequency.QuadPart) * 1000000;
}

#endif




/* USAGE:
 *
 * PerfVar xxx( "xxx" );
 * ...
 * xxx.BeginTiming();
 * ...
 * xxx.EndTiming();
 * ...
 * xxx.BeginTiming();
 * ...
 * xxx.EndTiming();
 * ...
 * xxx.Settle(); | xxx.Print(); | xxx.Dump();
 * xxx.Clean();
 * ...
 * xxx.BeginTiming();
 * ...
 * ...
 * ...
 */

class PerfVar 
{
public:
    // the statistics
    double mean;        // in micro-second
    double deviation;   // in micro-second
    double sum;         // in micro-second
    long long number;
    double max;
    double min;

    // the raw intermediate data
    bool isTiming;      // this is used ONLY under single thread environment
    double sum_square;
	double delay_episode;

    // the name of this variable
    const char* name;

public:
    inline PerfVar(const char* varName = NULL);

    inline void     BeginTiming();   // this is used ONLY under single thread environment
    inline double   EndTiming();  // this is used ONLY under single thread environment
	inline double	SuspendTiming();
	inline void		ResumeTiming();
	inline double   CheckTime();

    // when multi-thread enabled, more than one timing process can be performed concurrently.
    // so, for each timing thread, it has to store time stamps in its own TLS, and report the 
    // recorded time delay by AddRecord().
    inline long long	AddRecord(double recorded_delay);
    inline long long    Merge(const PerfVar& pv);

    inline int Clean();
    inline int Settle();

    inline int Dump(const char* fileName = NULL, bool auto_clean =false);
    inline int Print(FILE* fp = stdout, bool auto_clean =false);

private:
    LARGE_INTEGER startTime;    // this is used ONLY under single thread environment
    LARGE_INTEGER endTime;      // this is used ONLY under single thread environment
#if !defined(__linux__) && !defined(__APPLE__)
    LARGE_INTEGER frequency;    // the variable that records the hardware tick count frequency
#endif
};


PerfVar::PerfVar(const char* varName /*=NULL*/) 
    : mean(0), 
      deviation(-1), 
      sum(0), 
      number(0), 
      sum_square(0),
	  delay_episode(0),
      name(varName), 
      isTiming(false) 
{

#if !defined(__linux__) && !defined(__APPLE__)
    startTime.QuadPart = 0;
    endTime.QuadPart = 0;
    QueryPerformanceFrequency(&frequency);
#endif
}

void PerfVar::BeginTiming() 
{
    assert(isTiming == false);
    isTiming = true;
	delay_episode = 0;

    QueryPerformanceCounter(&startTime);
}

double PerfVar::SuspendTiming() 
{
    assert(isTiming == true);
    isTiming = false;

    QueryPerformanceCounter(&endTime);

#if !defined(__linux__) && !defined(__APPLE__)
    delay_episode += (endTime.QuadPart - startTime.QuadPart)
                    / (double) (frequency.QuadPart) * 1000000;
#else

    delay_episode += toMicroSeconds(endTime, startTime);
#endif
    return delay_episode;
}

void PerfVar::ResumeTiming() 
{
    assert(isTiming == false);
    isTiming = true;

    QueryPerformanceCounter(&startTime);
}

double PerfVar::EndTiming() 
{
    assert(isTiming == true);
    isTiming = false;

    QueryPerformanceCounter(&endTime);

#if !defined(__linux__) && !defined(__APPLE__)
    delay_episode += (endTime.QuadPart - startTime.QuadPart)
                    / (double) (frequency.QuadPart) * 1000000;

#else
    delay_episode += toMicroSeconds(endTime, startTime);
#endif
    AddRecord(delay_episode);

    return delay_episode;
}

double PerfVar::CheckTime() 
{
    assert(isTiming == true);

    QueryPerformanceCounter(&endTime);

#if !defined(__linux__) && !defined(__APPLE__)
    double delay_segment = (endTime.QuadPart - startTime.QuadPart)
                    / (double) (frequency.QuadPart) * 1000000;
#else
    double delay_segment = toMicroSeconds(endTime, startTime);
#endif

    return delay_episode + delay_segment;
}

long long PerfVar::AddRecord(double delay) 
{
    sum += delay;
    sum_square += delay * delay;
    max = (number == 0 || delay > max) ? delay : max;
    min = (number == 0 || delay < min) ? delay : min;
    number++;  
    return number;
}

long long PerfVar::Merge(const PerfVar& pv) 
{
    sum += pv.sum;
    sum_square += pv.sum_square;
    max = (pv.number > 0 && pv.max > max) ? pv.max : max;
    min = (pv.number > 0 && pv.min < min) ? pv.min : min;
    number += pv.number;  
    return number;
}

int PerfVar::Clean() 
{
    mean = 0;
    deviation = -1;
    sum = 0;
    number = 0;
    sum_square = 0;
	delay_episode = 0;
    max = 0;
    min = 0;
    return 0;
}

int PerfVar::Settle() 
{
    if (number != 0) 
    {
        mean = sum / number;
        deviation = sqrt(sum_square / number - mean * mean);
    }

    return 0;
}

int PerfVar::Dump(const char* fileName, bool auto_clean) 
{
    FILE* fp;
    const char* fName;

    Settle();

    if (fileName == NULL)
        fName = "./perf.tsv";
    else
        fName = fileName;

    fp = fopen(fName, "r");
    if (fp == NULL) 
    {
        fp = fopen(fName, "w");
        assert(fp);

        fprintf(fp, "VariableNume \t Mean(us) \t Dev.(us) \t Max(us) \t Min(us) \t Total(us) \t Number\n");
    } 
    else 
    {
        fclose(fp);
        fp = fopen(fName, "a");
        assert(fp);
    }

    if (name == NULL) 
    {
        fprintf(fp, "NONAME \t ");
    } 
    else 
    {
        fprintf(fp, "%s \t ", name);
    }

    fprintf(fp, "%.3lf \t %.3lf \t %.3lf \t %.3lf \t %.3lf \t %lld\n", 
        mean,
        deviation, 
        (number==0) ? 0 : max, 
        (number==0) ? 0 : min, 
        sum, 
        number);

    fclose(fp);
    
    if(auto_clean) Clean();

    return 0;
}

int PerfVar::Print(FILE* fp, bool auto_clean) 
{
    Settle();

    if (name == NULL) 
    {
        fprintf(fp, "NONAME \t ");
    } 
    else 
    {
        fprintf(fp, "%s \t ", name);
    }

    fprintf(fp, "mean=%.3lf  dev.=%.3lf  max=%.3lf min=%.3lf total=%.3lf  number=%lld\n",
        mean,
        deviation, 
        (number==0) ? 0 : max, 
        (number==0) ? 0 : min, 
        sum, 
        number);

    if(auto_clean) Clean();
    return 0;
}

void CPULoad(void* argv)
{
		double y = 0.0;
		double x = 1.0;

		while(1)
		{
			y += log(x);
			x = x+1.0;
		}
}

void ThroughputTest(int thread_num =1)
{
	FILE* fp = fopen("cputest.txt", "w");
	fprintf(fp, "#thread \t throughput (st) \t throughput (mt) \t speedup(%%)\n");
	fflush(fp);

	double throughput_st;

	for(int i=1; i<=thread_num; i++)
	{
		printf("#thread = %d\n", i);
		printf("---------------------------------\n");

		double x;
		double y;
		PerfVar throughput;

		for(int t=0; t<10; t++)
		{
			y = 0.0;
			x = 1.0;
			PerfVar timer;
			timer.BeginTiming();
			while(1)
			{
				y += log(x);

				double latency = timer.CheckTime();
		
				if(latency >= 5e6)
				{
					timer.EndTiming();
					throughput.AddRecord(x/5e6);
					throughput.Settle();
					printf("througput = %.1lf \t avg. = %.1lf \t y=%lf\n", x/5e6, throughput.mean, y);
					
					break;
				}
		
				x = x+1.0;
			}
		}
		printf("\n");

		throughput.Settle();
		if(i==1) 
			throughput_st = throughput.mean;

		fprintf(fp, "%d \t %.1lf \t %.1lf \t %.1lf%%\n", i, throughput.mean, throughput.mean*i, throughput.mean*i / throughput_st * 100.0);
		fflush(fp);
		throughput.Clean();

#if !defined(__linux__) && !defined(__APPLE__)
		_beginthread(CPULoad, 0, NULL);
#else
        std::thread cpuThread(CPULoad, (void*)NULL);
        cpuThread.detach();
#endif
	}
}


#endif
