#ifndef __MNIST_H__
#define __MNIST_H__

#include <vector>
#include <iostream>
#include <fstream>
#include <deque>
#include "utils.h"
#include "RL.h"
#include "image.h"
#include "DCNN.h"


#define MNIST_SIZE 28
#define SIZE_IMG (MNIST_SIZE * MNIST_SIZE)
#define SIZE_LABEL 10 // 0 ... 9, unknown

class MNIST_Data
{
public:
	unsigned char img[MNIST_SIZE][MNIST_SIZE];
	unsigned char label;
};


void DumpImage_MNIST_Training()
{
	std::ifstream file("MNIST/train-images.idx3-ubyte", std::ifstream::binary);

	assert(file.is_open());

	int magic_number;
	file.read((char*)&magic_number, sizeof(int)); magic_number = reverseInt(magic_number);
	assert(magic_number == 2051);

	int num_image;
	int row;
	int column;

	file.read((char*)&num_image, sizeof(int)); num_image = reverseInt(num_image);
	file.read((char*)&row, sizeof(int));	row = reverseInt(row);
	file.read((char*)&column, sizeof(int)); column = reverseInt(column);
	//printf("%d %d %d", num_image, row, column);

	int img_size = row*column;
	unsigned char* data = new unsigned char[img_size];

	for(int t=0; t<num_image; t++)
	{
		file.read((char*)data, img_size);
		printf("file %d, read %d bytes\n", t, file.gcount());
		assert(file);

		char filename[1000];
		sprintf(filename, "MNIST/training/train%d.bmp", t);

		Mat2BMP(data, row, column, filename);

		/*for(int i=0; i<row; i++)
		{
			for(int j=0; j<column; j++)
			{
				int tmp = data[i*column+j];
				printf("%d ", tmp);
			}
			printf("\n");
		}*/
	}

	delete[] data;
}


void DumpImage_MNIST_Testing()
{
	std::ifstream file("MNIST/t10k-images.idx3-ubyte", std::ifstream::binary);
	
	assert(file.is_open());

	int magic_number;
	file.read((char*)&magic_number, sizeof(int)); magic_number = reverseInt(magic_number);
	assert(magic_number == 2051);

	int num_image;
	int row;
	int column;

	file.read((char*)&num_image, sizeof(int)); num_image = reverseInt(num_image);
	file.read((char*)&row, sizeof(int));	row = reverseInt(row);
	file.read((char*)&column, sizeof(int)); column = reverseInt(column);
	//printf("%d %d %d", num_image, row, column);

	int img_size = row*column;
	unsigned char* data = new unsigned char[img_size];

	for(int t=0; t<num_image; t++)
	{
		file.read((char*)data, img_size);
		printf("file %d, read %d bytes\n", t, file.gcount());
		assert(file);

		char filename[1000];
		sprintf(filename, "MNIST/training/train%d.bmp", t);

		Mat2BMP(data, row, column, filename);

		/*for(int i=0; i<row; i++)
		{
			for(int j=0; j<column; j++)
			{
				int tmp = data[i*column+j];
				printf("%d ", tmp);
			}
			printf("\n");
		}*/
	}

	delete[] data;
}


void LoadData_MNIST(std::vector<MNIST_Data>& training_set, std::vector<MNIST_Data>& testing_set)
{	 
	int magic_number;
	int n_training;
	int n_testing;
	int row;
	int column;
	int tmp;

	std::ifstream file_training_img("MNIST/train-images.idx3-ubyte", std::ifstream::binary);
	assert(file_training_img.is_open());

	file_training_img.read((char*)&magic_number, sizeof(int)); magic_number = reverseInt(magic_number);
	assert(magic_number == 2051);

	file_training_img.read((char*)&n_training, sizeof(int)); n_training = reverseInt(n_training);
	file_training_img.read((char*)&row, sizeof(int));	row = reverseInt(row);
	file_training_img.read((char*)&column, sizeof(int)); column = reverseInt(column);
	//printf("%d %d %d", num_image, row, column);
	assert(row == column && row == MNIST_SIZE);


	std::ifstream file_training_label("MNIST/train-labels.idx1-ubyte", std::ifstream::binary);
	assert(file_training_label.is_open());

	file_training_label.read((char*)&magic_number, sizeof(int)); magic_number = reverseInt(magic_number);
	assert(magic_number == 2049);

	file_training_label.read((char*)&tmp, sizeof(int)); tmp = reverseInt(tmp);
	assert(tmp == n_training);

	training_set.resize(n_training);


	for(int t=0; t<n_training; t++)
	{
		file_training_img.read((char*)training_set[t].img, MNIST_SIZE*MNIST_SIZE);
		//printf("training image %d, read %d bytes. ", t, file_training_img.gcount());
		assert(file_training_img);
		assert(file_training_img.gcount() == MNIST_SIZE*MNIST_SIZE);

		file_training_label.read((char*)&(training_set[t].label), 1);
		//printf("training label %d, read %d bytes (l=%d).\n", t, file_training_label.gcount(), training_set[t].label);
		assert(file_training_label);
		assert(file_training_label.gcount() == 1);

		/*for(int i=0; i<row; i++)
		{
			for(int j=0; j<column; j++)
			{
				int tmp = training_set[t].img[i][j];
				printf("%3d ", tmp);
			}
			printf("\n");
		}*/

		if(t%10000 == 10000-1) printf("read %d training cases\n", t+1);
	}


	std::ifstream file_testing_img("MNIST/t10k-images.idx3-ubyte", std::ifstream::binary);
	assert(file_testing_img.is_open());

	file_testing_img.read((char*)&magic_number, sizeof(int)); magic_number = reverseInt(magic_number);
	assert(magic_number == 2051);

	file_testing_img.read((char*)&n_testing, sizeof(int)); n_testing = reverseInt(n_testing);
	file_testing_img.read((char*)&row, sizeof(int));	row = reverseInt(row);
	file_testing_img.read((char*)&column, sizeof(int)); column = reverseInt(column);
	//printf("%d %d %d", num_image, row, column);
	assert(row == column && row == MNIST_SIZE);


	std::ifstream file_testing_label("MNIST/t10k-labels.idx1-ubyte", std::ifstream::binary);
	assert(file_testing_label.is_open());

	file_testing_label.read((char*)&magic_number, sizeof(int)); magic_number = reverseInt(magic_number);
	assert(magic_number == 2049);

	file_testing_label.read((char*)&tmp, sizeof(int)); tmp = reverseInt(tmp);
	assert(tmp == n_testing);

	testing_set.resize(n_testing);


	for(int t=0; t<n_testing; t++)
	{
		file_testing_img.read((char*)testing_set[t].img, MNIST_SIZE*MNIST_SIZE);
		//printf("testing image %d, read %d bytes. ", t, file_testing_img.gcount());
		assert(file_testing_img);
		assert(file_testing_img.gcount() == MNIST_SIZE*MNIST_SIZE);

		file_testing_label.read((char*)&(testing_set[t].label), 1);
		//printf("testing label %d, read %d bytes (l=%d).\n", t, file_testing_label.gcount(), testing_set[t].label);
		assert(file_testing_label);
		assert(file_testing_label.gcount() == 1);

		/*for(int i=0; i<row; i++)
		{
			for(int j=0; j<column; j++)
			{
				int tmp = testing_set[t].img[i][j];
				printf("%3d ", tmp);
			}
			printf("\n");
		}*/

		if(t%10000 == 10000-1) printf("read %d testing cases\n", t+1);
	}

}





class Environment_Mnist: public Environment<SIZE_IMG, SIZE_LABEL>
{
public:
    std::vector<MNIST_Data> training_set;
	std::vector<MNIST_Data> testing_set;
    double groundtruth[SIZE_LABEL]; // the groundtruth label of the current image (indexed by 'data_id')
    
    struct LogItem
    {
        int50 data_id;
        int50 label;
        int50 action;
    };
    std::deque<LogItem> log;    // store the last 'log_capacity' log items
    int50 log_capacity;
    int50 timestamp;

protected:
	double* memory_x;
	double* memory_y;
	int50 num_sample;
    int50 data_id;
    int50 pass;

public:
	virtual ~Environment_Mnist()
	{
		delete[] memory_x;
		delete[] memory_y;	
	}

	Environment_Mnist(int50 log_size=32)
	{
		// load data
		LoadData_MNIST(training_set, testing_set);

		num_sample = training_set.size();
		memory_x = new double[SIZE_IMG * num_sample];
		memory_y = new double[SIZE_LABEL * num_sample];

		for(int t=0; t < num_sample; t++)
		{
			int offset = t*SIZE_IMG;
			for(int i=0; i<SIZE_IMG; i++)
				memory_x[offset+i] = (training_set[t].img[i/MNIST_SIZE][i%MNIST_SIZE])/(double)255.0;
		}
		for(int t=0; t < num_sample; t++)
		{
			int offset = t*SIZE_LABEL;
			for(int i=0; i<SIZE_LABEL; i++)
				memory_y[offset+i] = (training_set[t].label == i) ? 1.0 : 0.0;
		}	

		data_id = 0;
        pass = 1;
        log_capacity = log_size;
        timestamp = 0;
	}

    virtual void Reset() {}

    virtual bool Update(OUT_ double* x, OUT_ double& r, OUT_ bool& fTerminal, IN_ double* a)
    {
        // evaluate 'a'
        double* y = &memory_y[data_id * SIZE_LABEL];
		r = (vec_dist_L1(y, a, SIZE_LABEL) == 0) ? 1 : 0;
        fTerminal = true;

		// go to the the next image in the memory
		data_id = (data_id+1) % num_sample;
        if(data_id == 0) pass++;
        for(int i=0; i<SIZE_IMG; i++) x[i] = memory_x[data_id * SIZE_IMG + i];

        // logging
        y = &memory_y[data_id * SIZE_LABEL];
		for(int i=0; i<SIZE_LABEL; i++) groundtruth[i] = y[i];

        if(not log.empty()) log.back().action = vec_argmax<double>(a,SIZE_LABEL);
        while(log_capacity > 0 && log.size() > log_capacity) log.pop_front();
        log.push_back(LogItem());
        log.back().data_id = data_id;
        log.back().label = vec_argmax<double>(groundtruth, SIZE_LABEL);
        
        timestamp ++;

        return true;
    }
    
    virtual void Print(FILE* fp=stdout) 
    {
        Mat2BMP((unsigned char*)(training_set[data_id].img), MNIST_SIZE, MNIST_SIZE, "mnist_state.bmp");
    }
    
    virtual void Report(FILE* fp =stdout, int50 detail_level =1, bool reset =true) 
    {
        if(detail_level == 0)
        {
            fprintf(fp, " ENV._TIME \t pass | id \t");
        }
        else if(detail_level == 1)
        {
            fprintf(fp, "%7lld \t %4lld | %lld\t", timestamp, pass, data_id);
        }
        else if(detail_level == 2)
        {
            assert(log.size() <= log_capacity+1);
            if(log.size()<=1) return;
        
            fprintf(fp, "[MNIST Log] @ round %lld - %lld :\n", timestamp-(log.size()-1), timestamp-1);
            int50 nCorrect=0;
            for(int50 i=0; i<log.size()-1; i++) // do not report the last log item since it is (always) incomplete
            {
                nCorrect += (log[i].label==log[i].action) ? 1 : 0;
                if(log.size()-i<=32)
                {
                    fprintf(fp, "step #%lld \t img=%lld \t label=%lld \t output=%lld \t %s\n", timestamp-(log.size()-1)+i, log[i].data_id, log[i].label, log[i].action, (log[i].label==log[i].action)?"same":"diff");
                }
            }
            fprintf(fp, "accuracy in the last %lld steps  = %.2lf%%\n\n", log.size()-1, nCorrect/(double)(log.size()-1)*100);
        }
        else assert(0);  
    }
};



void mnist_test()
{
    static const int50 batch_size = 32;
    double learning_rate = 0.00025;
    double init_weight_range = 0.1;
    int50 num_parallel_learner = 2;
    char* filename_weight = NULL;
    
	// MNIST
    std::vector<DCNN_CONFIG> config;
	config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	1,	0,  0,  0, 28, 28));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		8,	5,  1, -2, 28, 28));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		16, 3,  2, -1, 14, 14));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		32, 3,  2, -1,  7,  7));
	config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		10, 7,  1,  0,  1,  1));

    int50 reporting_cycle = 100; // in epochs
    int50 nEpoch = 10000;

    // create the neural network
    DCNN<SIZE_IMG, SIZE_LABEL> nn(num_parallel_learner);
	nn.Setup(config);
	if(nn.size_output != SIZE_LABEL) {printf("I/O mismatch: \t %lld \t %lld\n", nn.size_output, SIZE_LABEL); getchar();}
	if(nn.size_input != SIZE_IMG) {printf("I/O mismatch: \t %lld \t %lld\n", nn.size_input, SIZE_IMG); getchar();}

	// initialize the weights
	if(filename_weight != NULL)
	{
		nn.LoadWeights(filename_weight);
	}
    else
    {
        std::mt19937 rng;
        rng.seed(12345);
        nn.RandomizeWeights(rng, init_weight_range);
    }

    // create the envionment
    Environment_Mnist env(batch_size*reporting_cycle);
    double x[SIZE_IMG] = {0};
    double r = 0;
    bool fTerminal = false;
    double a[SIZE_LABEL] = {0};

    double* data_x[batch_size]; for(int t=0; t<batch_size; t++) data_x[t] = new double[SIZE_IMG];
    double* data_y[batch_size]; for(int t=0; t<batch_size; t++) data_y[t] = new double[SIZE_LABEL];
    int50 data_action[batch_size];
    PerfVar pv_copy("Copy");

    
    env.Update(x, r, fTerminal, a);

    for(int50 epoch=1; epoch<=nEpoch; epoch++)
    {
        pv_copy.BeginTiming();
        double buffer_x[batch_size][SIZE_IMG];
        double buffer_y[batch_size][SIZE_LABEL];
        int50 buffer_action[batch_size];
        pv_copy.SuspendTiming();

        //// supervised learning mode
        //for(int t=0; t<batch_size; t++)
        //{
        //    //env.Print();
        //    nn.forward(x);
        //    int50 action = vec_argmax(nn.output, SIZE_LABEL);
        //    for(int i=0; i<SIZE_LABEL; i++) a[i] = 0; a[action] = 1;
        //    env.Update(x, r, fTerminal, a);

        //    for(int i=0; i<SIZE_IMG; i++) buffer_x[t][i] = x[i];
        //    for(int i=0; i<SIZE_LABEL; i++) buffer_y[t][i] = env.groundtruth[i]; 
        //}
        //nn.BatchUpdate(learning_rate, batch_size, buffer_x, buffer_y);

        // reinforcement learning mode (with greedy action policy)
        for(int t=0; t<batch_size; t++)
        {
            pv_copy.ResumeTiming();
            for(int i=0; i<SIZE_IMG; i++) buffer_x[t][i] = x[i];
            pv_copy.SuspendTiming();

            //env.Print();
            nn.forward(x);
            int50 action = vec_argmax(nn.output, SIZE_LABEL);
            for(int i=0; i<SIZE_LABEL; i++) a[i] = 0; a[action] = 1;
            env.Update(x, r, fTerminal, a);

            pv_copy.ResumeTiming();
            for(int i=0; i<SIZE_LABEL; i++) buffer_y[t][i] = a[i]*r;  
            buffer_action[t] = vec_argmax(a, SIZE_LABEL);
            pv_copy.SuspendTiming();
        }
        pv_copy.ResumeTiming();
        for(int t=0; t<batch_size; t++) for(int i=0; i<SIZE_IMG; i++) data_x[t][i] = buffer_x[t][i];
        for(int t=0; t<batch_size; t++) for(int i=0; i<SIZE_LABEL; i++) data_y[t][i] = buffer_y[t][i];
        for(int t=0; t<batch_size; t++) data_action[t] = buffer_action[t];
        pv_copy.EndTiming();

        //nn.BatchUpdate(learning_rate, batch_size, (double**)data_x, (double**)data_y, data_action);
        nn.MTBatchUpdate(learning_rate, batch_size, (double**)data_x, (double**)data_y, data_action);
        nn.MTFinishBatch(true);


        if(epoch%reporting_cycle==0)
        {
            nn.MTFinishBatch(true);

            printf("\nepoch %lld -------\n", epoch);
            env.Report();
            nn.Report();
            pv_copy.Print(); pv_copy.Clean();
        }
    }

    for(int t=0; t<batch_size; t++) {delete[] data_x[t]; delete[] data_y[t];}
    return;
}

#endif