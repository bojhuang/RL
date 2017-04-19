#pragma once


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>
#include <queue>
#include <vector>
#include <iostream>
#include <fstream>


#include "PerfVar.h"
#include "image.h"
#include "utils.h"


///////////////////////////////////////////////////////
///////////////////////////////////////////////////////


#if defined(__linux__) || defined(__APPLE__)

#include <chrono>
#include <thread>

inline void Sleep(int dwMilliseconds)
{
	std::chrono::milliseconds duration(dwMilliseconds);

	std::this_thread::sleep_for(duration);
}
#else
#include <process.h>
#endif


#define NTYPE_CONSTANT 0
#define NTYPE_RELU 1
#define NTYPE_SIGMOID 2
#define NTYPE_TANH 3
#define NTYPE_POOLING 4
#define NTYPE_SOFTMAX 5	   // currently only support softmax at the output layer
#define NTYPE_LINEAR 6

class Neuron
{
public:
	double f;
	double a;
	double g;
	double b;

	int50 type;
	int50 id;
	int50 id_layer;
	int50 id_row;
	int50 id_column;
	int50 id_height;



	void Setup(int50 t, int50 i, int50 l, int50 r, int50 c, int50 h)
	{
		type = t; id = i; id_layer = l; id_row = r; id_column = c; id_height = h;
	}

	double compute_f()
	{
		if(type == NTYPE_RELU)
		{
			f = (a > 0.0) ? a : 0.0;	
		}
		else if(type == NTYPE_SIGMOID)
		{
			f = 1 / (1 + exp(-a));
		}
		else if(type == NTYPE_SOFTMAX)
		{
			f = a; // leave real softmax computation outside this routine
		}
		else if(type == NTYPE_CONSTANT)
		{
			f = a;
		}
		else if(type == NTYPE_LINEAR)
		{
			f = a;
		}
		else
			assert(0);

		return f;
	}

	// g = b * f'(a)
	double compute_g()
	{
		if(type == NTYPE_RELU)
		{
			g = (a > 0.0) ? b : 0.0;
		}
		else if(type == NTYPE_CONSTANT)
		{
			g = 0.0;
		}
		else if(type == NTYPE_SIGMOID)
		{
			double f_prime = f / (1+ exp(a));
			g = b * f_prime;
		}
		else if(type == NTYPE_SOFTMAX)
		{
			g = b; // "psuedo-derivative" is 1
		}
		else if(type == NTYPE_LINEAR)
		{
			g = b;
		}
		else assert(0);

		return g;
	}
};


class DCNN_CONFIG
{
public:
	int50 H_FC;
	int50 L_FC;
	int50 W_FC;
	int50 S_FILTER;
	int50 STRIDE;
	int50 OFFSET[2];
	int50 type;

	DCNN_CONFIG(){}

	DCNN_CONFIG(int50 t, int50 h, int50 s, int50 stride, int50 offset, int50 l=0, int50 w=0)
		: type(t), H_FC(h), L_FC(l), W_FC(w), S_FILTER(s), STRIDE(stride) 
	{
		OFFSET[0] = OFFSET[1] = offset;
	}
};


template<int SIZE_INPUT, int SIZE_OUTPUT>
class DCNN
{
public:
	static const int50 MAX_DEPTH = 100;
	
	int50 DEPTH;
	DCNN_CONFIG config[MAX_DEPTH];
	int50 H_FC[MAX_DEPTH];
	int50 L_FC[MAX_DEPTH];
	int50 W_FC[MAX_DEPTH];
	int50 S_FILTER[MAX_DEPTH];
	int50 STRIDE[MAX_DEPTH];
	int50 OFFSET[MAX_DEPTH];


	int50 num_neuron;
	int50 num_weight;
	int50 num_flops;
	int50 size_input;
	int50 size_output;
	int50 offset_output;	// id of the first output neuron

	Neuron* neuron;
	double* weight;
    double* output;
	double* gradient_sum;   // for buffering the gradient sums over a mini-batch
    double* delta;	        // for buffering the weight delta
	double* aux1;	        // used by some SGD algorithms
	double* aux2;	        // used by some SGD algorithms
	//#define BUF_SIZE 50000000

protected:
	int50 idx_neuron[MAX_DEPTH];
	int50 idx_weight[MAX_DEPTH];
	bool f_destroy_neuron;
	bool f_destroy_weight;

    struct DCNNMTPackage
	{
        DCNN* self;
		int50 threadId;

		volatile int50 state;	
		#define MT_WAITING		0
		#define MT_BUSY_GRAD	1
		#define MT_BUSY_SGD		2 
		#define MT_FINISHED		3

		std::vector<double*> data_x;	// input data assigned to this thread
        std::vector<double*> data_y;	// label data assigned to this thread
        std::vector<int50> data_a;      // action data assigned to this thread
		DCNN* pShadow;				    // local NN of this thread
        double learning_rate;
        int50 batch_size;
		double mt_loss_sum;
		double mt_action_diff_sum;
	};
	DCNNMTPackage* mt_workspace;
    const int mt_nWorker;
	bool mt_in_progress;

    // logging facility
    PerfVar loss_l1;
    PerfVar accuracy;
    PerfVar pv_fp;
    PerfVar pv_bp;
    PerfVar pv_sgd;
    PerfVar pv_batch;

    // registers for the "error clip" trick 
    // (no member function facility is provided for this trick, and users need to directly edit these registers)
public:
    bool error_clip;
    double ERROR_MAX;
    double ERROR_MIN;

public:
	DCNN(int num_worker =0) 
		:   DEPTH(0), num_neuron(0), num_weight(0), 
            neuron(NULL), weight(NULL), output(NULL), delta(NULL), aux1(NULL), aux2(NULL), gradient_sum(NULL), f_destroy_neuron(false), f_destroy_weight(false), 
            mt_nWorker(num_worker), mt_workspace(NULL), mt_in_progress(false),
            pv_fp("FP"), pv_bp("BP"), pv_sgd("SGD"), pv_batch("Batch"), error_clip(false)
	{
		//printf("DCNN \t %p\n", &f_destroy_neuron);
	}

	DCNN(const std::vector<DCNN_CONFIG>& configuration, Neuron* neuron_list=NULL, double* weight_list=NULL, bool error_clip=false, int num_worker =0) 
		:   DEPTH(0), num_neuron(0), num_weight(0),
            neuron(NULL), weight(NULL), output(NULL), delta(NULL), aux1(NULL), aux2(NULL), gradient_sum(NULL), f_destroy_neuron(false), f_destroy_weight(false), 
            mt_nWorker(num_worker), mt_workspace(NULL), mt_in_progress(false),
            pv_fp("FP"), pv_bp("BP"), pv_sgd("SGD"), pv_batch("Batch"), error_clip(false)
	{
		Setup(configuration, neuron_list, weight_list, error_clip);
	}

	virtual ~DCNN()
	{
		//printf("~DCNN \t %p\n", &f_destroy_neuron);
		if(f_destroy_neuron == true) delete[] neuron;
		if(f_destroy_weight == true) delete[] weight;
		if(gradient_sum != NULL) delete[] gradient_sum;
		if(aux1 != NULL) delete[] aux1;
		if(aux2 != NULL) delete[] aux2;
        if(delta != NULL) delete[] delta;
        if(output != NULL) delete[] output;

        if(mt_workspace != NULL)
        {
            for(int i=0; i<mt_nWorker; i++) delete mt_workspace[i].pShadow;
            delete[] mt_workspace;
        }
	}

	// TODO: disable copy construction and operator=

	// user don't have to specify L_FC and W_FC, which will be computed based on filter settings
	// if user do specify L_FC and W_FC, the routine will double check if they are consistent with the filter settings
	bool Setup(const std::vector<DCNN_CONFIG>& configuration, Neuron* neuron_list=NULL, double* weight_list=NULL, bool fErrorClip=false)
	{
		DEPTH = configuration.size();
		for(int50 l=0; l<DEPTH; l++)
		{
			config[l]=configuration[l];
		}

		for(int50 l=0; l<DEPTH; l++)
		{
			H_FC[l] = config[l].H_FC;
			L_FC[l] = config[l].L_FC;
			W_FC[l] = config[l].W_FC;
			S_FILTER[l] = config[l].S_FILTER;
			STRIDE[l] = config[l].STRIDE;
			OFFSET[l] = config[l].OFFSET[0];
		}

		for(int50 l=1; l<DEPTH; l++)
		{
			int50 L_tmp = config[l].L_FC;
			int50 W_tmp = config[l].W_FC;

			config[l].L_FC = 0;
			int50 off_tail;
			for(off_tail = config[l].OFFSET[0]+config[l].S_FILTER-1; off_tail <= config[l-1].L_FC-1-config[l].OFFSET[0]; off_tail += config[l].STRIDE)
			{
				if(off_tail < 0)
				{
					printf("[network auto-config error] some filter is completely out of range.\n");
					getchar();
					return false;
				}
				config[l].L_FC++;	
			}
			if(config[l].L_FC == 0)
			{
				printf("[network auto-config error] feature map is too small to hold a single filter.\n");
				getchar();
				return false;
			}

			if(off_tail - config[l].STRIDE < config[l-1].L_FC-1)
			{
				printf("[network auto-config warning] some cell is not covered by any filter.\n");
				getchar();
				return false;
			}

			config[l].W_FC = 0;
			for(off_tail = config[l].OFFSET[1]+config[l].S_FILTER-1; off_tail <= config[l-1].W_FC-1-config[l].OFFSET[1]; off_tail += config[l].STRIDE)
			{
				if(off_tail < 0)
				{
					printf("[network auto-config error] some filter is completely out of range.\n");
					getchar();
					return false;
				}
				config[l].W_FC++;	
			}
			if(config[l].W_FC == 0)
			{
				printf("[network auto-config error] feature map is too small to hold a single filter.\n");
				getchar();
				return false;
			}
			if(off_tail - config[l].STRIDE < config[l-1].W_FC-1)
			{
				printf("[network auto-config warning] some cell is not covered by any filter.\n");
				getchar();
				return false;
			}

			if(config[l].STRIDE > config[l].S_FILTER)
			{
				printf("[network auto-config warning] some cell is not covered by any filter.\n");
				getchar();
				return false;
			}

			if(L_tmp != 0 && L_tmp != config[l].L_FC)
			{
				printf("[network auto-config error] parameter mismatch at layer %lld, L_FC=%lld, L_tmp=%lld\n", l, config[l].L_FC, L_tmp);
				getchar();
				return false;
			}

			if(W_tmp != 0 && W_tmp != config[l].W_FC)
			{
				printf("[network auto-config error] parameter mismatch at layer %lld, W_FC=%lld, W_tmp=%lld\n", l, config[l].W_FC, W_tmp);
				getchar();
				return false;
			}
			
			/*if(S_FILTER[l]%2 == 1)
			{
				if((L_FC[l-1] -1 -2*OFFSET[l][0]) % STRIDE[l] != 0 || (L_FC[l-1] -1 -2*OFFSET[l][0]) < 0)
				{
					printf("[error] filter misarrangement in length: layer=%lld, L_FC[%lld]=%lld, offset_len=%lld, stride=%lld\n", l, l-1, L_FC[l-1], OFFSET[l][0], STRIDE[l]);
					getchar();
				}

				if((W_FC[l-1] -1 -2*OFFSET[l][1]) % STRIDE[l] != 0 || (W_FC[l-1] -1 -2*OFFSET[l][1]) < 0)
				{
					printf("[error] filter misarrangement in width: layer=%lld, W_FC[%lld]=%lld, offset_wid=%lld, stride=%lld\n", l, l-1, W_FC[l-1], OFFSET[l][1], STRIDE[l]);
					getchar();
				}

				if(OFFSET[l][0] > S_FILTER[l]/2 || OFFSET[l][1] > S_FILTER[l]/2)
				{
					printf("[error] feature map uncovered: layer=%lld, offset=(%lld,%lld), filter_size=%lld\n", l, OFFSET[l][0], OFFSET[l][1], S_FILTER[l]);
					getchar();
				}

				L_FC[l] = (L_FC[l-1] -1 -2*OFFSET[l][0]) / STRIDE[l] + 1;
				W_FC[l] = (W_FC[l-1] -1 -2*OFFSET[l][1]) / STRIDE[l] + 1;
			}*/
		} //.~ l

		num_neuron = 0;
		for(int50 l=0; l<DEPTH; l++)
		{
			idx_neuron[l] = num_neuron;
			num_neuron += config[l].L_FC * config[l].W_FC * config[l].H_FC;
		}
		idx_neuron[DEPTH] = num_neuron;

		num_weight = 0;
		for(int50 l=1; l<DEPTH; l++)
		{
			idx_weight[l] = num_weight;
			num_weight += config[l].H_FC * (config[l].S_FILTER * config[l].S_FILTER * config[l-1].H_FC + 1);
		}
		idx_weight[DEPTH] = num_weight;

		size_input = idx_neuron[1];
		size_output = num_neuron - idx_neuron[DEPTH-1];
		offset_output = idx_neuron[DEPTH-1];

		// reset the vectors of the neurons, weights, and deltas
		if(f_destroy_neuron == true) delete[] neuron; // destroy if neuron is a self-constructed list
		if(neuron_list != NULL) 
		{
			neuron = neuron_list;
			f_destroy_neuron = false;
		}
		else
		{
			neuron = new Neuron[num_neuron];
			f_destroy_neuron = true;
		}

		if(f_destroy_weight == true) delete[] weight; // destroy if weight is a self-constructed list
		if(weight_list != NULL) 
		{
			weight = weight_list;
			f_destroy_weight = false;
		}
		else
		{
			weight = new double[num_weight];
			f_destroy_weight = true;
		}

        // gradient_sum, delta, output, and the aux buffers are always self-constructed lists, so will need to be destroyed before resetting
        if(gradient_sum != NULL) delete[] gradient_sum; 
		gradient_sum = new double[num_weight];
        for(int50 i=0; i<num_weight; i++) gradient_sum[i] = 0.0;
		
		if(aux1 != NULL) delete[] aux1;
		aux1 = new double[num_weight];
		for(int50 i=0; i<num_weight; i++) aux1[i] = 0.0;

		if(aux2 != NULL) delete[] aux2;
		aux2 = new double[num_weight];
		for(int50 i=0; i<num_weight; i++) aux2[i] = 0.0;

        if(delta != NULL) delete[] delta;
		delta = new double[num_weight];
        for(int50 i=0; i<num_weight; i++) delta[i] = 0.0;

        if(output != NULL) delete[] output;
		output = new double[size_output];
        for(int50 i=0; i<size_output; i++) output[i] = 0.0;

		int50 id = 0;
		for(int50 l=0; l<DEPTH; l++)
		{
			for(int50 r=0; r<config[l].L_FC; r++)
			{
				for(int50 c=0; c<config[l].W_FC; c++)
				{
					for(int50 h=0; h<config[l].H_FC; h++)
					{
						assert(id == nid(l,r,c,h));

						neuron[id].Setup(config[l].type, id, l, r, c, h);
					
						id++;
					}
				}
			}
		}

        // setup the workspace for MT learning 
        if(mt_nWorker > 0)
        {
		    mt_workspace = new DCNNMTPackage[mt_nWorker];
		    mt_in_progress = false;

		    for(int50 tid=0; tid<mt_nWorker; tid++)
		    {	
                mt_workspace[tid].self = this;
			    mt_workspace[tid].threadId = tid;
			    mt_workspace[tid].state = MT_WAITING;
			    mt_workspace[tid].data_x.clear();
			    mt_workspace[tid].data_y.clear();
                mt_workspace[tid].data_a.clear();
                mt_workspace[tid].mt_loss_sum = 0;
                mt_workspace[tid].mt_action_diff_sum = 0;
                mt_workspace[tid].pShadow = new DCNN(configuration, NULL, weight, 0); //all neural networks share the same weight vector, which is maintained in the leader thread
			    mt_workspace[tid].mt_loss_sum = 0;
			    mt_workspace[tid].mt_action_diff_sum = 0;
		    }
		    //SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        }

        // error clip setup
        error_clip = fErrorClip;
        for(int tid=0; tid<mt_nWorker; tid++) mt_workspace[tid].pShadow->error_clip = fErrorClip;

		// estimate GFlops of the specified network (forward, 1/6)
		num_flops = 0;
		for(int50 l=1; l<DEPTH; l++) 
		{
			num_flops += (config[l].H_FC * config[l].L_FC * config[l].W_FC) * (config[l-1].H_FC * config[l].S_FILTER * config[l].S_FILTER);
		}

		printf("#neuron = %lld, #weight = %lld, #FLOPS = %lld(x6)\n", num_neuron, num_weight, num_flops);

		return true;
	}

	bool LoadWeights(const char* filename = "weight.bin")
	{
		std::ifstream file_weight(filename, std::ifstream::binary);
		if(file_weight.is_open() == false)
		{
			printf("[error] cannot open %s\n",filename);
			getchar();
			return false;
		}

		file_weight.read((char*)weight, sizeof(double)*num_weight);
		
		if(file_weight.gcount() != sizeof(double)*num_weight)
		{
			printf("[error] failed to read %lld bytes from %s\n", sizeof(double)*num_weight, filename);
			getchar();

			file_weight.close();
			return false;
		}

		char buf[1];
		file_weight.read(buf, 1);
		if(file_weight.gcount() > 0)
		{
			printf("[error] %s (size= %lld bytes) is larger than a correct weight files\n", filename, sizeof(double)*num_weight);
			getchar();

			file_weight.close();
			return false;
		}


		file_weight.close();
		return true;
	}

    bool RandomizeWeights(std::mt19937& rng, double magnitude, bool random_bias=false)
    {
	    std::uniform_real_distribution<double> weight_generator(-magnitude/2, magnitude/2);

		int50 id = 0;
		for(int50 l=1; l<DEPTH; l++)
		{
			for(int50 h=0; h<config[l].H_FC; h++)
			{
				for(int50 i=0; i<config[l].S_FILTER; i++)
					for(int50 j=0; j<config[l].S_FILTER; j++)
						for(int50 k=0; k<config[l-1].H_FC; k++)
						{
							assert(id == wid(l,h,i,j,k));
							weight[id] = weight_generator(rng);
							if(id < 100) printf("nn weight[%lld] = %lf\n", id, weight[id]);
							id++;
						}

				assert(id == wid(l,h,config[l].S_FILTER, 0, 0));

				weight[id] = (random_bias) ? weight_generator(rng) : magnitude; 
				if(id<100) printf("nn weight[%lld] = %lf\n", id, weight[id]);
				id++;
			}
		}
		
		double weight_avg_l1 = vec_avg_L1(weight, num_weight);
		printf("avg. values of initial weights (L1) = %lf\n", weight_avg_l1);
        return true;
    }

	bool DumpWeights(const char* filename = "weight.bin")
	{
		std::ofstream file_weight(filename, std::ifstream::binary);
		if(file_weight.is_open() == false)
		{
			printf("[error] cannot open %s\n", filename);
			getchar();
			return false;
		}

		file_weight.write((char*)weight, sizeof(double)*num_weight);
		file_weight.close();
		return true;
	}


public:
	inline int50 nid(int50 l, int50 r, int50 c, int50 h)
	{
		return idx_neuron[l] + (r * W_FC[l] + c) * H_FC[l] + h; 
	}

	// the constant term is indexed by (l, h, S, 0, 0)
	inline int50 wid(int50 l, int50 h, int50 i, int50 j, int50 k)
	{
		return idx_weight[l] + h * (S_FILTER[l] * S_FILTER[l] * H_FC[l-1] + 1) +
				(i * S_FILTER[l] + j) * H_FC[l-1] + k;
	}

	// results are buffered in the 'output' vector
	void forward(const double* x)
	{
        pv_fp.BeginTiming();

		for(int50 id=0; id<size_input; id++)
		{
			neuron[id].f = x[id];
		}

		for(int50 id = idx_neuron[1]; id < num_neuron; id++)
		{
			int50 l = neuron[id].id_layer;
			int50 r = neuron[id].id_row;
			int50 c = neuron[id].id_column;
			int50 h = neuron[id].id_height;
			assert(id == nid(l,r,c,h) && id == neuron[id].id);

			neuron[id].a = 0.0;

			int50 id_w = wid(l, h, 0, 0, 0);
			int50 offset_i = OFFSET[l] + r * STRIDE[l];

			for(int50 i=0; i<S_FILTER[l]; i++, offset_i++)
			{
				// need to reset offset_j and re-compute id_n when changing row in the receptive field
				int50 offset_j = OFFSET[l] + c * STRIDE[l];
				int50 id_n = nid(l-1, offset_i, offset_j, 0);

				for(int50 j=0; j<S_FILTER[l]; j++, offset_j++)
				{
					// zero padding for "out-of-range neurons"
					if( offset_i < 0 || offset_i >= L_FC[l-1] || 
						offset_j < 0 || offset_j >= W_FC[l-1] )
					{
						id_n += H_FC[l-1];
						id_w += H_FC[l-1];
						continue;
					}

					for(int50 k=0; k<H_FC[l-1]; k++)
					{
						//assert(offset_i < 0 || offset_i >= config[l-1].L_FC || 
						//	   offset_j < 0 || offset_j >= config[l-1].W_FC || 
						//	   id_n == nid(l-1, offset_i, offset_j, k));
						//assert(id_w == wid(l,h,i,j,k));
						
						double x = neuron[id_n].f; 
						double w = weight[id_w];
						neuron[id].a += w * x;

						id_n ++;
						id_w ++;
					}
				}
			}

			// add the constant term
			assert(id_w == wid(l,h,S_FILTER[l], 0, 0));
			neuron[id].a += weight[id_w];

			neuron[id].compute_f();

			// layer-wise normalization for softmax neurons
			if(neuron[id].type == NTYPE_SOFTMAX && id == idx_neuron[l+1]-1)
			{
				double a_max = neuron[offset_output].a;
				for(int50 i=idx_neuron[l]; i<=id; i++) 
				{
					a_max = (neuron[i].a > a_max) ? neuron[i].a : a_max;
				}
				
				double z = 0.0;
				for(int50 i=idx_neuron[l]; i<=id; i++)
				{
					neuron[i].f = exp(neuron[i].a - a_max);	   // normalize a to avoid zero denominator
					z += neuron[i].f;
				}

				for(int50 i=idx_neuron[l]; i<=id; i++)
				{
					neuron[i].f = neuron[i].f / z;
				}
			}
		}

		for(int50 i=0; i<size_output; i++) output[i] = neuron[offset_output+i].f;

        pv_fp.EndTiming();
	}

    // bp computation w.r.t. the MLE loss for softmax outputs, or the one-half-MSE loss for all other type of output neurons
    // the results are accumulated into the 'gradient_sum' vector
	void backward(const double* y)
	{
        pv_bp.BeginTiming();

		for(int50 id = 0; id < num_neuron; id++) neuron[id].b = 0.0;

		// the first term in the product chain is (f-y) for the MSE loss if the real derivative is used in compute_g() later, 
		// OR the first term is still (f-y) for the MLE loss if the psuedo-drivative is used for the softmax outputs.
		for(int50 i=0; i<size_output; i++)
		{
			int50 id = idx_neuron[DEPTH-1]+i;
			neuron[id].b = neuron[id].f - y[i];

            #define ERROR_MAX 1.0
            #define ERROR_MIN -1.0
            if(error_clip) 
                neuron[id].b = (neuron[id].b > ERROR_MAX) ? (ERROR_MAX) : (neuron[id].b < ERROR_MIN) ? (ERROR_MIN) : neuron[id].b;
		}

		for(int50 id = num_neuron-1; id >= idx_neuron[1]; id--)
		{
			int50 l = neuron[id].id_layer;
			int50 r = neuron[id].id_row;
			int50 c = neuron[id].id_column;
			int50 h = neuron[id].id_height;
			assert(id == nid(l,r,c,h) && id == neuron[id].id);

			neuron[id].compute_g();
			if(neuron[id].g == 0.0) continue;

			int50 id_w = wid(l, h, 0, 0, 0);
			int50 offset_i = OFFSET[l] + r * STRIDE[l];
			
			for(int50 i=0; i<S_FILTER[l]; i++, offset_i++)
			{
				// need to reset offset_j and re-compute id_n when changing row in the receptive field
				int50 offset_j = OFFSET[l] + c * STRIDE[l];
				int50 id_n = nid(l-1, offset_i, offset_j, 0);

				for(int50 j=0; j<S_FILTER[l]; j++, offset_j++)
				{
					// ignore the "out-of-range neurons"
					if( offset_i < 0 || offset_i >= L_FC[l-1] || 
						offset_j < 0 || offset_j >= W_FC[l-1] )
					{
						id_n += H_FC[l-1];
						id_w += H_FC[l-1];
						continue;
					}

					for(int50 k=0; k<H_FC[l-1]; k++)
					{
						//assert(offset_i < 0 || offset_i >= config[l-1].L_FC || 
						//	   offset_j < 0 || offset_j >= config[l-1].W_FC || 
						//	   id_n == nid(l-1, offset_i, offset_j, k));
						//assert(id_w == wid(l,h,i,j,k));

						double x = neuron[id_n].f; 
						double w = weight[id_w];
						neuron[id_n].b += neuron[id].g * w;
						gradient_sum[id_w] += neuron[id].g * x; 

						id_n ++;
						id_w ++;
					}
				}
			}

			// update the weight of the constant term
			assert(id_w == wid(l,h,S_FILTER[l], 0, 0));
			gradient_sum[id_w] += neuron[id].g;
		}

        pv_bp.EndTiming();
	}

    // given a data point, compute the gradient, with the results accumulated into the 'gradient_sum' vector;
    // 'y' is a supervised-learning feedback if 'action' is -1, or is a reinforcement-learning feedback if 'action' is a valid index of the output,
    // in the latter case 'y' may be filled with the nn output by the routine;
    // this is a sub-routine of the mini-batch framework, which does not take charge of clearing the gradient_sum vector beforehand
	void ComputeGradient(const double* x, double* y, const int50 action =-1)
	{
		forward(x);

        if(action >= 0)
        {
            assert(action<size_output); 
            // construct a "groundtruth output" based on the rl feedback, which are different from the nn output only at the 'action' position
            for(int50 i=0; i<size_output; i++) 
            {
                y[i] = (i==action) ? (y[i]) : (output[i]);     
            }
        }
            
		backward(y);

        //logging
        double loss;
        double correct;

        loss = vec_dist_L1(y, output, size_output);
        correct = (vec_argmax(y,size_output) == vec_argmax(output, size_output)) ? 1 : 0;
        
        loss_l1.AddRecord(loss);
        accuracy.AddRecord(correct);
	}

    // given a gradient-sum vector (by default using itselves gradients), update the [wid_start, wid_last] segment of 
    // the weight delta (as well as all the auxiliary data) and of the weight vector if 'direct_update' is true; 
	// this is a sub-routine of the mini-batch framework, so it does not take charge of computing/clearing the gradient vector
	void SGD(double learning_rate, int50 batch_size, bool direct_update =true, int50 wid_start =-1, int50 wid_last =-1, const double* gradient =NULL)
	{
		if(gradient==NULL)  gradient = gradient_sum;
        if(wid_start==-1) wid_start = 0;
        if(wid_last==-1) wid_last = num_weight-1;
		
		// gradient descent
		for(int50 id=wid_start; id <= wid_last; id++)
		{
			double g = gradient[id] / batch_size;
			double alpha = learning_rate;
			

			//// vanilla SGD
			//delta[id] = -alpha * g;

			//// momentum-based SGD
			//double gamma = 0.9;
			//double* momentum = aux1;
			//momentum[id] = gamma * momentum[id] + alpha * g;
			//delta[id] = -momentum[id];

			// RMSprop
			static const double gamma = 0.95;
			static const double epsilon = 1e-8;
			double* e_g2 = aux1;
			e_g2[id] = gamma * e_g2[id] + (1-gamma) *g *g;
			double rms_g = sqrt(e_g2[id]+epsilon);
			delta[id] = -alpha * g / rms_g;

			//// AdaDelta
			//double gamma = 0.95;
			//double epsilon = 1e-6;
			//double* e_g2 = aux1;
			//double* e_dx2 = aux2;
			//e_g2[id] = gamma * e_g2[id] + (1-gamma) * g * g;
			//double rms_g = sqrt(e_g2[id]+epsilon);
			//double rms_dx = sqrt(e_dx2[id]+epsilon);
			//double dx = - rms_dx / rms_g * g;
			//e_dx2[id] = gamma * e_dx2[id] + (1-gamma) * dx * dx;
			//delta[id] = dx;

			//// Adam
			//double gamma_1 = 0.9;
			//double gamma_2 = 0.99;
			//double epsilon = 1e-8;
			//double* e_g = aux1;
			//double* e_g2 = aux2;
			//e_g[id] = gamma_1 * e_g[id] + (1-gamma_1) * g;
			//e_g2[id] = gamma_2 * e_g2[id] + (1-gamma_2) * g * g;
			//delta[id] = -alpha * (e_g[id]/(1-gamma_1)) / (sqrt(e_g2[id]/(1-gamma_2))+epsilon);


            weight[id] = (direct_update) ? (weight[id]+delta[id]) : (weight[id]);
		}
    }

    // perform a single-threaded mini-batch update. 'y' is supervised-learning feedback if 'action' is NULL, and is reinforcement-learning feedback otherwise
    void BatchUpdate(double learning_rate, int50 batch_size, double* x[], double* y[], int50 action[] =NULL)
    {
        pv_batch.BeginTiming();

        // clear the gradient vector at the beginning of a mini-batch
		memset(gradient_sum, 0, num_weight*sizeof(double));
		//for(int50 i = 0; i < num_weight; i++) gradient_sum[i] = 0.0;
		
		for(int50 t=0; t<batch_size; t++)
		{
            ComputeGradient(x[t], y[t], (action)?(action[t]):(-1));
        }

        pv_sgd.BeginTiming();
        SGD(learning_rate, batch_size);
        pv_sgd.EndTiming();

        pv_batch.EndTiming();
    }


    // multi-threading routines
public:
    // IMPORTANT: this is a nonblocking routine, so we need to make sure that the training data is protected while the parallel learners work on it.
    bool MTBatchUpdate(double learning_rate, int50 batch_size, double* x[], double* y[], int50 action[] =NULL)
    {
        // MT checking
        if(mt_nWorker <= 0) throw std::invalid_argument("[DCNN Error] MTStartBatch called on a DCNN with no parallel worker.\n");  
        //for(int50 i=0; i<mt_nWorker; i++)
		//{
		//	if(mt_workspace[i].state != MT_WAITING || mt_workspace[i].data_x.empty() != true)
		//	{
		//		printf("[MT Error] worker %lld is not ready to work: \t state=%lld \t nTask=%lld\n", i, mt_workspace[i].state, mt_workspace[i].data_x.size());
		//		throw std::invalid_argument("[MT Error] some worker is not ready to work.\n");
		//	}
		//}
        //if(mt_in_progress == true) throw std::invalid_argument("[DCNN Error] mt_in_progress = true when launching a new MT batch.\n");
  
        MTFinishBatch(true);
        mt_in_progress = true;
        pv_batch.BeginTiming();

		// let the main thread take charge of assigning the learning tasks out 
		// (so that the behavior of multi-threaded learning is exactly the same with the single-threaded version)
		int50 tid = 0;
		for(int50 t=0; t<batch_size; t++)
		{
			mt_workspace[tid].data_x.push_back(x[t]);
            mt_workspace[tid].data_y.push_back(y[t]);
            if(action) mt_workspace[tid].data_a.push_back(action[t]);
			tid = (tid+1) % mt_nWorker;
		}

		// signal all worker threads
		for(int50 tid=0; tid<mt_nWorker; tid++)
		{
            mt_workspace[tid].learning_rate = learning_rate;
            mt_workspace[tid].batch_size = batch_size;
			mt_workspace[tid].state = MT_BUSY_GRAD;

#if !defined(__linux__) && !defined(__APPLE__)
		    _beginthread(MTWorkerMain, 0, &(mt_workspace[tid]));
#else
		    std::thread threadObj(MTWorkerMain, &(mt_workspace[tid]));
		    threadObj.detach();
#endif
			
		}

        return true;
    }

    // try to finish up a learning batch if there exists one; 
    // will wait for the parallel workers if in blocked mode, otherwise will return false for incomplete batch
    bool MTFinishBatch(bool blocked)
    {
        if(mt_nWorker <= 0) return true;

        if(mt_in_progress == true)
		{			
			// check if everyone has completed their own works
			bool allDone;
			while(1)
			{
				allDone = true;
				for(int50 tid=0; tid<mt_nWorker; tid++)
				{
					int50 state = mt_workspace[tid].state;

					if(state != MT_FINISHED)
					{
						allDone = false;
						break;
					}
				}

				if(allDone == true || blocked == false) break;
				
				//printf("[DCNN info] waiting for a learning batch to complete...\n");
				Sleep(5);	
			}

			if(allDone == false)
			{
				assert(blocked == false);
				return false;
			}

			// finish up the mini-batch by finally updating the weight vector of the global NN (see MTWorkerMain() for more details), 
            // aggregating the logging results, and resetting the workers
			for(int50 i=0; i<num_weight; i++) weight[i] += delta[i];
            
            // debug
            //printf("w = %e \t d = %e \t g = %e \t g1 = %e \t g2 = %e\n", vec_avg_L1(weight,num_weight), vec_avg_L1(delta,num_weight), vec_avg_L1(gradient_sum,num_weight), vec_avg_L1(mt_workspace[0].pShadow->gradient_sum,num_weight), vec_avg_L1(mt_workspace[1].pShadow->gradient_sum,num_weight));

            for(int50 tid=0; tid<mt_nWorker; tid++)
			{
				assert(mt_workspace[tid].state == MT_FINISHED);

				//loss_sum += mt_workspace[tid].mt_loss_sum;
				//action_diff_sum += worker[i].mt_action_diff_sum;
				//n_since_last_report += worker[i].idx.size();						
				
				mt_workspace[tid].mt_loss_sum = 0;
				mt_workspace[tid].mt_action_diff_sum = 0;
				mt_workspace[tid].data_x.clear();
                mt_workspace[tid].data_y.clear();
                mt_workspace[tid].data_a.clear();
				mt_workspace[tid].state = MT_WAITING;
			}

			mt_in_progress = false;
            pv_batch.EndTiming();
		}

        return true;
    }

protected:
    // Compute the (partial) gradients over the data points assigned to this thread, using the local NN of this thread;
	//  the result is also stored in the Local NN of this thread (in 'worker[i].nn.delta').
	// Worker[0] is further responsible to aggregate the partial gradients into a delta vector;
	//  to avoid any multi-threading synchronization overhead, worker[0] only computes a weight-delta vector and stores it 
	//  in the delta vector of the Global NN (in 'nn.delta'), and the weight vector needs to be updated somewhere else in order to complete the SGD update.
	static void __cdecl MTWorkerMain(void* pArgv)
	{
		DCNNMTPackage* pWorkspace = (DCNNMTPackage*)pArgv;
        DCNN* self = pWorkspace->self;
		int50 tid = pWorkspace->threadId;
        int50 nWorker = self->mt_nWorker;
		std::vector<double*>& data_x = pWorkspace->data_x;
        std::vector<double*>& data_y = pWorkspace->data_y;
        std::vector<int50>& data_a = pWorkspace->data_a;
		DCNN& nn_local = *(pWorkspace->pShadow);

		while(pWorkspace->state != MT_BUSY_GRAD) ;
        assert(data_x.size() == data_y.size());
			
		// do its own work
		memset(nn_local.gradient_sum, 0, nn_local.num_weight*sizeof(double));
		//for(int50 i = 0; i < nn_local.num_weight; i++) nn_local.gradient_sum[i] = 0.0;
		for(int50 t=0; t<data_x.size(); t++)
		{
            nn_local.ComputeGradient(data_x[t], data_y[t], (not data_a.empty())?(data_a[t]):(-1));

			// logging
			//double nn_output[SIZE_ACTION]; for(int50 i=0; i<SIZE_ACTION; i++) nn_output[i] = nn_local.neuron[nn_local.offset_output+i].f;
			//int50 action_rpm = vec_argmax<double>(self->rpm[data_id].a, SIZE_ACTION);
			//int50 action_now = vec_argmax<double>(nn_output, SIZE_ACTION);
			//double r_rpm = self->rpm[data_id].r_lt;
			//double r_nn = nn_output[action_rpm];
			//double loss_l1 = abs(r_rpm - r_nn);
			//self->worker[tid].mt_loss_sum += loss_l1;
			//self->worker[tid].mt_action_diff_sum += (action_rpm == action_now) ? 1 : 0;
            
            // debug
            //printf("%lld \t %lld \t %e \t %e \t %lld \t %e \t %e\n", tid, t, vec_avg_L1(data_x[t],nn_local.size_input), vec_avg_L1(data_y[t],nn_local.size_output), data_a[t], vec_avg_L1(nn_local.weight, nn_local.num_weight), vec_avg_L1(nn_local.gradient_sum, nn_local.num_weight));
		}
		pWorkspace->state = MT_BUSY_SGD;

		// wait for everyone to complete their works, as a semi-spin barrier (i.e. using Sleep(0))
		while(1)
		{
			bool allDone = true;

			for(int50 i=0; i<nWorker; i++)
			{
				int50 state = self->mt_workspace[i].state;

				if(state == MT_BUSY_GRAD) // the worker will proceed if other threads are in MT_WAITING -- this is by design!
				{
					allDone = false;
					break;
				}
			}

			if(allDone == true) break;
			Sleep(0);	   
		}



		// aggregate all the local gradients to the gradient_sum vector of the global NN, and then compute the weight delta using SGD;
        // each worker takes charge of only a designated vector segment so that they can work in parallel here 
		int50 wid_first = nn_local.num_weight / nWorker * tid;
		int50 wid_last =  (tid == nWorker-1) ? (nn_local.num_weight -1) : (nn_local.num_weight / nWorker * (tid+1) - 1);
		//printf("%lld \t %lld \t %lld\n", tid, wid_first, wid_last);

        for(int j=wid_first; j<=wid_last; j++) self->gradient_sum[j]  = 0;
		for(int50 i=0; i<nWorker; i++)
		{
			const DCNN& nn_remote = *(self->mt_workspace[i].pShadow);
			for(int j=wid_first; j<=wid_last; j++) 
				self->gradient_sum[j] += nn_remote.gradient_sum[j];	
		}

		if(pWorkspace->batch_size > 0)
		{
            nn_local.pv_sgd.BeginTiming();
			self->SGD(pWorkspace->learning_rate, pWorkspace->batch_size, false, wid_first, wid_last);
            nn_local.pv_sgd.EndTiming();
		}

		pWorkspace->state = MT_FINISHED;
	} 


    // auxiliary routines
public:
    // one-half MSE
	double Loss_MSE(const double* y)
	{
		double loss = 0.0;
		for(int50 i=0; i<size_output; i++)
		{
			loss += (neuron[i+offset_output].f - y[i]) * (neuron[i+offset_output].f - y[i]) / 2.0;
		}
		return loss;
	}

	double Loss_MLE(const double* y)
	{
		double loss = 0.0;
		for(int50 i=0; i<size_output; i++)
		{
			loss += -y[i] * log(neuron[i+offset_output].f);
		}
		return loss;
	}

	// verify if the current delta[] is the correct gradient (for the last neuron in the model)
	// typically used with batch_size = 1
	bool CheckGradient(const double* x, const double* y)
	{
		printf("---verifying gradients: ");
		double loss = (config[DEPTH-1].type == NTYPE_SOFTMAX) ? Loss_MLE(y) : Loss_MSE(y);
		
		double step = 1e-6;
		
		for(int50 id = 0; id < num_weight; id++)
		{
			weight[id] -= step;

			forward(x);
			double loss_new = (config[DEPTH-1].type == NTYPE_SOFTMAX) ? Loss_MLE(y) : Loss_MSE(y);
			double pd = (loss - loss_new)/step;

			//printf("%lld \t %e \t %e\n", id, delta[id], pd);
			weight[id] += step;

			if(pd > 1e-300 && abs(delta[id] - pd)/pd > 0.1) 
			{
				printf("found difference in gradient. id=%lld \t delta=%e \t pd=%e \n", id, delta[id], pd);
				//getchar();
				//return false;
			}

			if(id % ((num_weight-1)/10) == 0) printf("%lld0%% ", id/((num_weight-1)/10));
		}

		printf("--------\n");
		return true;
	}

	void PrintFilters_BP(int50 l, int50 r, int50 c, int50 h)
	{
		for(int50 id = 0; id < num_neuron; id++) neuron[id].b = 0.0;

		neuron[nid(l,r,c,h)].b = 1.0;

		for(int50 id = nid(l,r,c,h); id >= idx_neuron[1]; id--)
		{
			int50 l = neuron[id].id_layer;
			int50 r = neuron[id].id_row;
			int50 c = neuron[id].id_column;
			int50 h = neuron[id].id_height;
			assert(id == nid(l,r,c,h) && id == neuron[id].id);

			int50 offset_i = config[l].OFFSET[0] + r * config[l].STRIDE;
			int50 offset_j = config[l].OFFSET[1] + c * config[l].STRIDE;
			int50 id_n = nid(l-1, offset_i, offset_j, 0);
			int50 id_w = wid(l, h, 0, 0, 0);

			for(int50 i=0; i<config[l].S_FILTER; i++, offset_i++)
			{
				// need to reset offset_j and re-compute id_n when changing row in the receptive field
				offset_j = config[l].OFFSET[1] + c * config[l].STRIDE;
				id_n = nid(l-1, offset_i, offset_j, 0);

				for(int50 j=0; j<config[l].S_FILTER; j++, offset_j++)
				{
					for(int50 k=0; k<config[l-1].H_FC; k++)
					{
						assert(offset_i < 0 || offset_i >= config[l-1].L_FC || 
							   offset_j < 0 || offset_j >= config[l-1].W_FC || 
							   id_n == nid(l-1, offset_i, offset_j, k));

						assert(id_w == wid(l,h,i,j,k));

						if(offset_i >= 0 && offset_i < config[l-1].L_FC && offset_j >= 0 && offset_j < config[l-1].W_FC)
						{
							double w = weight[id_w];

							neuron[id_n].b += neuron[id].b * w;
						}
						//else
						//{
						//	printf("PrintFilters_BP out of range.");
						//	getchar();
						//}

						id_n ++;
						id_w ++;
					}
				}
			}
		}

	}

	void PrintFilters(bool quiet_mode=false)
	{
		if(config[0].H_FC != 1)
		{
			printf("currently only support grey-level input.");
			//getchar();
			return;
		}

		unsigned char* m = new unsigned char[size_input];

		for(int50 l=1; l<DEPTH; l++)
		{
			for(int50 h=0; h<config[l].H_FC; h++)
			{
				PrintFilters_BP(l, config[l].L_FC/2, config[l].W_FC/2, h);

				double b_min = neuron[0].b;
				for(int50 i=0; i<idx_neuron[1]; i++) b_min = min(b_min, neuron[i].b);

				double b_max = neuron[0].b;
				for(int50 i=0; i<idx_neuron[1]; i++) b_max = max(b_max, neuron[i].b);

				for(int50 i=0; i<size_input; i++)
				{
					int50 v = (b_max == b_min) ? 0 : (int50)((neuron[i].b - b_min)/(b_max - b_min) * 255);
					m[i] = v;
				}

				char filename[1000];
				sprintf(filename, "filter_%lld_%lld.bmp", l, h);

				Mat2BMP(m, config[0].L_FC, config[0].W_FC, filename);
				if (quiet_mode == false) printf("filter (l=%lld , h=%lld) printed to %s\n", l, h, filename);
			}
		}

		delete[] m;
	}

    virtual void Report(FILE* fp=stdout, int50 timestamp =-1, int50 detail_level=1, bool reset=true)
    {
        if(detail_level == 0)
        {
            fprintf(fp, " DCNN_EPOCH \t weight_avg \t delta_avg \t L1 \t acc. \t T_fp \t T_bp \t T_sgd \tT_batch\t");
        }
        else
        {
            double delta_avg = vec_avg_L1(delta, num_weight);
            double delta_lmax = vec_norm_Lmax(delta, num_weight);
            double delta_l0 = vec_norm_L0(delta, num_weight, 0.01*delta_avg);

            double weight_avg = vec_avg_L1(weight, num_weight);
            double weight_lmax = vec_norm_Lmax(weight, num_weight);
            double weight_l0 = vec_norm_L0(weight, num_weight, 0.01*weight_avg);

            // merge-and-clean the logs of the shadow NNs
            for(int i=0; i<mt_nWorker; i++)
            {
                loss_l1.Merge(mt_workspace[i].pShadow->loss_l1); mt_workspace[i].pShadow->loss_l1.Clean();    
                accuracy.Merge(mt_workspace[i].pShadow->accuracy); mt_workspace[i].pShadow->accuracy.Clean();
                pv_fp.Merge(mt_workspace[i].pShadow->pv_fp); mt_workspace[i].pShadow->pv_fp.Clean();
                pv_bp.Merge(mt_workspace[i].pShadow->pv_bp); mt_workspace[i].pShadow->pv_bp.Clean();
                pv_sgd.Merge(mt_workspace[i].pShadow->pv_sgd); mt_workspace[i].pShadow->pv_sgd.Clean();
                pv_batch.Merge(mt_workspace[i].pShadow->pv_batch); mt_workspace[i].pShadow->pv_batch.Clean();
            }
            loss_l1.Settle(); 
            accuracy.Settle();
            pv_fp.Settle();
            pv_bp.Settle();
            pv_sgd.Settle();
            pv_batch.Settle();

            if(detail_level == 1)
            {
                fprintf(fp, "%7lld \t %.2e \t %.2e \t %.3lf \t %.1lf%% \t %.1lf \t %.1lf \t %.1lf \t %.1lf \t", 
                    timestamp,
                    vec_avg_L1(weight, num_weight),
                    vec_avg_L1(delta, num_weight),
                    loss_l1.mean,
                    accuracy.mean * 100,
                    pv_fp.mean /1000,
                    pv_bp.mean /1000,
                    pv_sgd.mean /1000,
                    pv_batch.mean /1000
                );    
            }
            else if (detail_level == 2)
            {
                fprintf(fp, "[DCNN Log] @ epoch %lld\n", timestamp);
                fprintf(fp, "DCNN weight: %.2e (avg. L1) \t %.2e (Lmax) \t %.0lf (L0 >1%% avg.)\n", weight_avg, weight_lmax, weight_l0);
                fprintf(fp, "DCNN delta : %.2e (avg. L1) \t %.2e (Lmax) \t %.0lf (L0 >1%% avg.)\n", delta_avg, delta_lmax, delta_l0);
                fprintf(fp, "avg. perf. : %.3lf (L1 loss) \t %.2lf%% (accuracy)\n", loss_l1.mean, accuracy.mean * 100);
            
                pv_fp.Print(fp);
                pv_bp.Print(fp);
                pv_sgd.Print(fp); 
                pv_batch.Print(fp); 
            }
            else assert(0);

            if(reset)
            {
                loss_l1.Clean(); 
                accuracy.Clean();
                pv_fp.Clean();
                pv_bp.Clean();
                pv_sgd.Clean();
                pv_batch.Clean();
            }
        }

    }
};
