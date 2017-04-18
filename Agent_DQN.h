#ifndef __AGENT_DQN_H__
#define __AGENT_DQN_H__


#include "DCNN.h"
#include "RL.h"





template<int50 SIZE_PERCEPT, int50 SIZE_ACTION>
class Agent_DQN: public Agent<SIZE_PERCEPT, SIZE_ACTION>
{
//// parameters -------------------------
//
// actor parameters
#define DEFAULT_RPM_MIN_SIZE		1000
#define DEFAULT_RPM_MAX_SIZE		2000
#define DEFAULT_EPSILON_MAX			1.0
#define DEFAULT_EPSILON_MIN	        0.1

// learner parameters
#define DEFAULT_BATCH_SIZE			4
#define INIT_WEIGHT_RANGE           0.1
#define LEARNING_RATE		        0.00025 // the current DCNN class uses one-half MSE, so we need to double the learning rate

#define UPDATE_INTERVAL				4
#define UPDATE_INTERVAL_EVALUATION	10000
#define TD_DISCOUNT					0.99
#define TD_MAX_GAP					4
#define ERROR_CLIP                  true

// system-level parameters
#define DEFAULT_NUM_WORKER			2
#define RPM_COMPRESSION
#define MT_ACTOR
//#define DUMP_TRAINING_DATA

protected:
    // actor parameters
    const int50 RPM_MIN_SIZE;
	const int50 RPM_MAX_SIZE;
    double epsilon_max;
	double epsilon_min;
    int50 annealing_start;
	int50 annealing_length;

	// learner parameters
	const int50 BATCH_SIZE;
	double init_weight_range;
	double learning_rate;

	// system-level parameters
	int50 NUM_WORKER;


//// core data structures -------------------------
//
public:
	DCNN<SIZE_PERCEPT, SIZE_ACTION> nn;        // used as action evaluation function in decision making (i.e. Q^{T.t})
	DCNN<SIZE_PERCEPT, SIZE_ACTION> nn_target; // used as action evaluation function in TD updates (i.e. Q^T)


	// quantize the [0,1]-valued percept x to a discrete value in {0...255};
	// the quantization divides [0,1] into 256 intervals, 
	// with compressed state 0 corresponding to the range [0 , 0.5/255), which is of length 0.5/255;
	// the compressed state 255 corresponding to the range [254.5/255 , 1], which is of length 0.5/255;
	// and any other compressed state 0 < x_c < 255 corresponding to ranges [(x_c-0.5)/255 , (x_c+0.5)/255), which is of length 1/255.
	//
	// PROPERTY: given any real number x in [0,1], let x_c in {0...255} be the compressed state of x, and x_cr in [0,1] be the real number restored from x_c, 
	//           although x_cr is inevictably different from x in general (as we loss 7-byte of information during the compression),
	// 			 the method guarantees that x and x_cr always "correspond" to the same compressed state, for any such x,
	//           i.e., we have Encode(x) = Encode(x_cr) = x_c for any x, where x_cr = Decode(Encode(x)).
	//
	// IMPORTANT: the compressed state x_c, although ranged in {0...255}, should NOT be used as a 256-level pixel value. 
	//            this is because the quantizing method used here is DIFFERENT from the popular way (which is used in EnvironmentALE) 
	//			  to map between a 256-level pixel value and a real number in [0,1). 
	//			  For example, consider using the compressed state x_c directly as the pixel value and convert it to a new input x' to the DCNN agent,
	//			  which will then further compress the new input to a new compressed state x'_c, then x_c != x'_c when the pixel value is >128.
	//			  So, the compressed state is "simply" the compressed state used to save memory, it is not a pixel value.
	class CompressedMemItem
	{
	public:
		unsigned char x_c[SIZE_PERCEPT];
		double a[SIZE_ACTION];
		double r_lt;
		int50 step;
		double r_st;

		inline static unsigned char Encode(double x) 
		{
			if(x<0.0 || x>1.0) throw std::invalid_argument("[RL Error] input out of range when compressing rpm\n");
			return (int)round(x*255.0);
		}

		inline static double Decode(unsigned char x_c)
		{
			return (x_c / (double)255.0);
		}

        CompressedMemItem(){}

		CompressedMemItem(const unsigned char* xx_c, const double* aa =NULL, const double rr =0.0, const int50 ss =0, const double rr_st =0)
		{
			if(xx_c!=NULL) for(int50 i=0; i<SIZE_PERCEPT; i++) x_c[i] = xx_c[i];
			if(aa!=NULL) for(int50 i=0; i<SIZE_ACTION; i++) a[i] = aa[i];
			r_lt = rr;
			step = ss;
			r_st = rr_st;
		}

		CompressedMemItem(const double* xx, const double* aa =NULL, const double rr =0.0, const int50 ss =0, const double rr_st =0)
		{
			if(xx!=NULL) 
			{
				for(int50 i=0; i<SIZE_PERCEPT; i++) 
					x_c[i] = Encode(xx[i]);
			}
			if(aa!=NULL) for(int50 i=0; i<SIZE_ACTION; i++) a[i] = aa[i];
			r_lt = rr;	
			step = ss;
			r_st = rr_st;
		}

		CompressedMemItem(const typename Agent<SIZE_PERCEPT, SIZE_ACTION>::MemItem& item) : CompressedMemItem(item.x, item.a, item.r_lt, item.step, item.r_st) {}

        void CopyToBuffer (OUT_ double* buffer_x_cr, OUT_ double * buffer_a, OUT_ double& buffer_r, OUT_ int50& buffer_step, OUT_ double& buffer_rst) const
        {
            for(int50 i=0; i<SIZE_PERCEPT; i++) buffer_x_cr[i] = Decode(x_c[i]);
            for(int50 i=0; i<SIZE_ACTION; i++) buffer_a[i] = a[i];
            buffer_r = r_lt;
            buffer_step = step;
            buffer_rst = r_st;
        }

	private:
		void Test()
		{
			printf("%lld\n", (unsigned char)round(0.998*256));
			printf("%lld\n", (unsigned char)round(0.999*256));

			for(int50 i=0; i<=255; i++)
			{
				double x = i/(double)255;
				unsigned char x_c = (int)round(x*255.0);
				double x_cr = (x_c / (double)255.0);
				printf("%3d \t\n", i);
				if(x != x_cr) 
					printf("%lf \t %lf\n\n", x, x_cr);
		
			}

			for(int50 pixel=0; pixel<=255;pixel++)
			{
				double x = pixel/256.0;
				unsigned char x_c = (int)round(x*255.0);
		
				unsigned char  x_crc;
				for(int50 t=0; t<10; t++)
				{
					double x_r = x_c/(double)255.0;
					x_crc = (int)round(x_r*255.0);
				}

				if(pixel!=x_c || x_c!=x_crc) 
				{
					printf("%lld \t %lld \t %lld\n", pixel, x_c, x_crc);
				}
			}


			//for(int50 r=0; r<256; r++)
			//for(int50 g=0; g<256; g++)
			//for(int50 b=0; b<256; b++)
			for(int50 p=0; p<256; p++)
			{
					int50 r,g,b;
					r=g=b=p;
				double x = ((r * 0.2989) +(g * 0.5870) + (b * 0.1140)) / 256;
				unsigned char x_compressed = round(x*256);
				double x_restored = x_compressed/(double)256.0;

				printf("%lf%% \t %lf \t %lf \t %e\n", 100*(x-x_restored)/x, x, x_restored, x-x_restored);
				if(100*(x-x_restored)/x > 0.1/*%*/) getchar();
				//if(x_restored!=x) throw std::invalid_argument("wrong");
			}
		}
	};
#ifdef RPM_COMPRESSION
	Queue<CompressedMemItem> rpm;
#else
	Queue<MemItem> rpm;
#endif
	
    // buffer for the batch training data
    double** batch_x;
    double** batch_y;
    int50* batch_action;

	std::mt19937 rng;
    int50 timestamp;
    int50 learning_epoch;
	int50 learning_round;


//// multi-threading facility -------------------------
//
	volatile bool mt_action_done;
	double* mt_action_x;
	double* mt_action_a;


//// logging facility -------------------------------------
//
public:
	PerfVar learner_timer;
	PerfVar actor_timer;
    PerfVar mt_timer;
    PerfVar dump_weight_timer;
	PerfVar actor_activeness;
	//FILE* fp_log;
	//double* w_bak;					// buffer for the weights at the end of the last reporting epoch
	//int50  n_since_last_report;



public:
	Agent_DQN(  std::vector<DCNN_CONFIG>& config,
                char* filename_weight =NULL,
                int50 arg_batch_size =DEFAULT_BATCH_SIZE,
                int50 arg_num_worker =DEFAULT_NUM_WORKER,
                int50 arg_rpm_min_size =DEFAULT_RPM_MIN_SIZE, 
				int50 arg_rpm_max_size =DEFAULT_RPM_MAX_SIZE, 
                double arg_epsilon_max =DEFAULT_EPSILON_MAX,
				double arg_epsilon_min =DEFAULT_EPSILON_MIN)
		:   
            RPM_MIN_SIZE(arg_rpm_min_size),         // actor parameters
		    RPM_MAX_SIZE(arg_rpm_max_size),
            epsilon_max(arg_epsilon_max),
            epsilon_min(arg_epsilon_min),
            annealing_start(RPM_MIN_SIZE),
		    annealing_length(RPM_MAX_SIZE),           
            BATCH_SIZE(arg_batch_size),             // learner parameters
            init_weight_range(INIT_WEIGHT_RANGE),
            learning_rate(LEARNING_RATE),           
		    NUM_WORKER(arg_num_worker),             // system parameters
		    learner_timer("Learner"), 
            actor_timer("Actor  "), 
            mt_timer("MT_Overhead"),
            actor_activeness("Activeness"),   
            nn(arg_num_worker),
            rpm(RPM_MAX_SIZE),
            timestamp(0)
	{
		if(RPM_MIN_SIZE > RPM_MAX_SIZE) throw std::invalid_argument("[RL Error] RPM_MIN_SIZE > RPM_MAX_SIZE\n");

        rng.seed(12345);

		nn.Setup(config, NULL, NULL, ERROR_CLIP);
		if(nn.size_output != SIZE_ACTION) {printf("I/O mismatch: \t %lld \t %lld\n", nn.size_output, SIZE_ACTION); getchar();}
		if(nn.size_input != SIZE_PERCEPT) {printf("I/O mismatch: \t %lld \t %lld\n", nn.size_input, SIZE_PERCEPT); getchar();}

		// initialize the weights
		if(filename_weight != NULL)
		{
			nn.LoadWeights(filename_weight);
			//nn.PrintFilters();
		}
		else
		{
			nn.RandomizeWeights(rng, init_weight_range);
		}

		// make nn_target a copy of nn
		nn_target.Setup(config);
		memcpy(nn_target.weight, nn.weight, nn.num_weight*sizeof(double));

        // allocate space for the batch data buffers
        batch_x = new double*[BATCH_SIZE];
        batch_y = new double*[BATCH_SIZE];
        batch_action = new int50[BATCH_SIZE];
        for(int t=0; t<BATCH_SIZE; t++) 
        {
            batch_x[t] = new double[SIZE_PERCEPT];
            batch_y[t] = new double[SIZE_ACTION];
        }

        // MT setup
		mt_action_done = true;

		// initialize the logging facilities
		//w_bak = new double[nn.num_weight]; // for logging
		//n_since_last_report = 0;
		learning_epoch = 0;
		learning_round = 0; 
        dump_weight_timer.BeginTiming();
	}

    virtual ~Agent_DQN()
	{
        for(int t=0; t<BATCH_SIZE; t++)
        {
            delete[] batch_x[t];
            delete[] batch_y[t];
        }
        delete[] batch_x;
        delete[] batch_y;
        delete[] batch_action;

		nn.DumpWeights();			
		//nn.PrintFilters(true);
		//delete[] w_bak;
		//fclose(fp_log);
	}


public:
    // perform a batch learning with training data sampled from rpm (and thus the data in the function arguments are actually not used)
	virtual bool Learn(IN_ double* x, IN_ double* a, IN_ double r)
	{
        assert(x == NULL && a == NULL);
        if(BATCH_SIZE <= 0) return true;

        if(NUM_WORKER > 0)
        {
            nn.MTFinishBatch(true);
        }

        mt_timer.BeginTiming();
        // sample data from rpm to construct the batch data
		for(int50 t=0; t<BATCH_SIZE; t++)
		{
			std::uniform_int_distribution<int50> rpm_sampler(0, rpm.size()-1);
			int50 data_id = rpm_sampler(rng);
            
            MemItem item;
            rpm[data_id].CopyToBuffer(batch_x[t], item.a, item.r_lt, item.step, item.r_st);           
            int50 action = vec_argmax(item.a, SIZE_ACTION);
            batch_action[t] = action;
            for(int50 i=0; i<SIZE_ACTION; i++) batch_y[t][i] = (i==action) ? item.r_lt : 0; 

            //printf("training data: step=%lld \t action=%lld \t r_lt=%lf \t r_st=%lf\n", item.step, action, item.r_lt, item.r_st);

#ifdef DUMP_TRAINING_DATA
			// logging
			if(learning_epoch % epoch_reporting_gap == 0)
			{
				if(t==0) 
					system("del /Q batch\\*.bmp");
				#ifdef RPM_COMPRESSION
				double x[SIZE_PERCEPT];
				rpm[data_id].SetPercept(x);
				#else
				double* x = rpm[data_id].x;
				#endif
				double* a = rpm[data_id].a;
				double r = rpm[data_id].r_lt;
				DumpTrainingImgs(x, a, r, t, "batch");
			}
#endif
		}

        learning_epoch++;
		learning_round += BATCH_SIZE; 
        mt_timer.EndTiming();

        if(NUM_WORKER > 0) // MT learning mode
        {
			// start a new learning epoch so that the weight delta can be computed in parallel in the next 'UPDATE_INTERVAL' time steps 
            nn.MTBatchUpdate(learning_rate, BATCH_SIZE, batch_x, batch_y, batch_action);
        }
        else // ST learning mode
        {            
            nn.BatchUpdate(learning_rate, BATCH_SIZE, batch_x, batch_y, batch_action);  
        }

		return true;
	}

	virtual bool Act(OUT_ double* a, IN_ double* x)
	{
		int50 action;

		// epsilon-greedy
		double epsilon = epsilon_max - (epsilon_max-epsilon_min)/(double)annealing_length * (double)(timestamp - annealing_start);
		epsilon = (epsilon > epsilon_max) ? epsilon_max : (epsilon < epsilon_min) ? epsilon_min : epsilon;
		std::uniform_real_distribution<double> u_0_1(0.0, 1.0);
        double p = u_0_1(rng);
		if(p < epsilon) 
		{
			action = floor(u_0_1(rng)*(double)SIZE_ACTION); // u_0_1 \in [0,1)
			actor_activeness.AddRecord(0);
		}
		else
		{
			nn.forward(x);
			action = vec_argmax<double>(nn.output, SIZE_ACTION); 	
			actor_activeness.AddRecord(1);
		}

		for(int50 i=0; i<SIZE_ACTION; i++) a[i] = 0.0;
		a[action] = 1.0;

		return true;
	}

	virtual bool TakeAction(OUT_ double* a, IN_ double* x, IN_ double r, IN_ bool fTerminal=false)
	{		
		actor_timer.BeginTiming();
#ifdef MT_ACTOR
		if(mt_action_done == false) throw std::invalid_argument("[RL Error] mt_action synchroization error\n");
		mt_action_x = (double*)x;
		mt_action_a = a;
		mt_action_done = false;
		_beginthread(MTActorMain, 0, this);
#else 
        Act(a, x); 
#endif

		// update the target evaluation function, short-term memory and replay memory
		if(timestamp % UPDATE_INTERVAL_EVALUATION == 0 && rpm.size() >= RPM_MIN_SIZE) 
        { 
            memcpy(nn_target.weight, nn.weight, nn.num_weight*sizeof(double));
        }
		if(not stm.empty())
		{
			double r_discounted = r;
			for(int50 t=stm.size()-1; t>=0; t--)
			{
				stm[t].r_st += r_discounted;
				r_discounted = r_discounted * TD_DISCOUNT;
			}
		}
		if(fTerminal || stm.size() >= TD_MAX_GAP)
		{
			int50 stm_new_size = (fTerminal) ? (0) : (TD_MAX_GAP-1);
			assert(stm.size() >= stm_new_size);
			assert(stm.size() <= TD_MAX_GAP);
			
			// compute the tail reward
			double r_tail = 0;
			if(not fTerminal)
			{
				// Q-Learning evaluation function
				nn_target.forward(x);
				r_tail = vec_max<double>(nn_target.output, SIZE_ACTION);
			}
		
			while(stm.size() > stm_new_size)
			{
				// append the tail reward to the long-term return
				stm.front().r_lt = stm.front().r_st + pow(TD_DISCOUNT, stm.size()) * r_tail;

                // old rpm items are automatically removed when rpm is full;
                // note that this is MT-safe (only) because the main thread has been responsible for copying batch data from rpm to dedicated buffers
				rpm.push_back(stm.front(), true);
				stm.pop_front();
			}
		}

#ifdef MT_ACTOR
		// we have to finish the actor thread here, because the following code block may do a SGD update which
		// will change nn.weight, on which the actor is depending
		while(mt_action_done == false) 
		{
			Sleep(0);
		}
#endif
		actor_timer.SuspendTiming();


		// update long-term memory (option 2: do one mini-batch update every UPDATE_INTERVAL rounds via the replay memory)
		if(timestamp % UPDATE_INTERVAL == 0 && rpm.size() >= RPM_MIN_SIZE)
		{	
            Learn(NULL, NULL, 0);
		} 

		// add the latest experience to the short-term memory
		actor_timer.ResumeTiming();
		stm.emplace_back(x, a, 0.0, timestamp, 0.0);
		actor_timer.EndTiming();

        timestamp ++;

		return true;
	}


public:
	static void __cdecl MTActorMain(void* pArgv)
	{
		//SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
		Agent_DQN<SIZE_PERCEPT,SIZE_ACTION>* self = (Agent_DQN<SIZE_PERCEPT,SIZE_ACTION>*)pArgv;
		if(self->mt_action_done == true) throw std::invalid_argument("[RL Error] mt_action synchroization error\n");
		
		self->Act(self->mt_action_a, self->mt_action_x);
		self->mt_action_done = true;
	}

 
    virtual void Report(FILE* fp=stdout, int50 detail_level=1, bool reset=true)
    {
        if(detail_level == 0)
        {
            fprintf(fp, "AGENT_T\t" " active\t" "T_actor\t" "   T_mt\t");
            nn.Report(fp, learning_epoch, 0, reset);
        }
        else if(detail_level == 1)
        {
            actor_activeness.Settle();
            actor_timer.Settle();
            mt_timer.Settle();
            fprintf(fp, "%7lld\t" "%7.3lf\t" "%7.1lf\t" "%7.1lf\t", timestamp, actor_activeness.mean, actor_timer.mean/1000, mt_timer.mean/1000);
            
            nn.MTFinishBatch(true);
            nn.Report(fp, learning_epoch, 1, reset);

            if(reset)
            {
                actor_activeness.Clean();
                actor_timer.Clean();
                mt_timer.Clean();
            }
        }
        else if(detail_level == 2)
        {
            fprintf(fp, "[DQN Log] @ agent timestamp %lld\n", timestamp);
            actor_timer.Print(fp, reset);
            actor_activeness.Print(fp, reset); 
            learner_timer.Print(fp, reset); 
            mt_timer.Print(fp, reset); 
            fprintf(fp, "\n");

            nn.MTFinishBatch(true);
            nn.Report(fp, learning_epoch, 2, reset);
            fprintf(fp, "\n");

            static const double dump_weight_cycle = 300; // in seconds
            if(dump_weight_timer.CheckTime() > dump_weight_cycle * 1e6) 
		    {
                nn.DumpWeights();
			    //nn.PrintFilters(true);

			    dump_weight_timer.EndTiming();
			    dump_weight_timer.BeginTiming();
		    }
        }
        else assert(0);
    }


  //  virtual void Report(FILE* fp=stdout) 
  //  {
  //      // logging
		//if(learning_epoch % epoch_reporting_gap == 0)
		//{
		//	Logging();	
		//}
  //
  //      // logging
		//if(learning_epoch % epoch_reporting_gap == 0)
		//{
		//	FILE* fp_dcnn_log = fopen("dcnn_log.txt", "a");
		//	fprintf(fp_dcnn_log, "\nepoch %lld\n", learning_epoch);
		//	fclose(fp_dcnn_log);
		//}
   //
  //      // logging
		//	double nn_output[SIZE_ACTION]; for(int50 i=0; i<SIZE_ACTION; i++) nn_output[i] = nn_local.neuron[nn_local.offset_output+i].f;
		//	int50 action_rpm = vec_argmax<double>(self->rpm[data_id].a, SIZE_ACTION);
		//	int50 action_now = vec_argmax<double>(nn_output, SIZE_ACTION);
		//	double r_rpm = self->rpm[data_id].r_lt;
		//	double r_nn = nn_output[action_rpm];
		//	double loss_l1 = abs(r_rpm - r_nn);
		//	self->worker[tid].mt_loss_sum += loss_l1;
		//	self->worker[tid].mt_action_diff_sum += (action_rpm == action_now) ? 1 : 0;
        //
		//	// logging: report the result of every data point with known groundtruth if we are in a "reporting epoch"
		//	if(self->learning_epoch % self->epoch_reporting_gap == 0) 
		//	{
		//		fprintf(stdout, "d%2lld: r_rpm = %lf \t r_nn = %lf \t loss=%lf \t a_now = %2lld (%s) \t (a=%lld , r=%.1lf , t=%lld)\n", 
		//			t*(self->NUM_WORKER)+tid,
		//			r_rpm, 
		//			r_nn, 
		//			loss_l1,
		//			action_now,
		//			((action_rpm == action_now) ? "same" : "diff"),
		//			action_rpm, self->rpm[data_id].r_st, self->rpm[data_id].step);
      //
		//		FILE* fp_dcnn_log = fopen("dcnn_log.txt", "a");
		//		fprintf(fp_dcnn_log, "data %2lld: r_rpm = %lf \t r_nn = %lf \t loss=%lf \t a_now = %2lld (%s) \t (a=%lld , r=%.1lf , t=%lld)\n", 
		//			t*(self->NUM_WORKER)+tid,
		//			r_rpm, 
		//			r_nn, 
		//			loss_l1,
		//			action_now,
		//			((action_rpm == action_now) ? "same" : "diff"),
		//			action_rpm, self->rpm[data_id].r_st, self->rpm[data_id].step);
		//		fclose(fp_dcnn_log);
		//	}
  //  }
  //
 //    void Logging()
	//{
	//	if(dump_weight_timer.CheckTime() > weight_dumping_gap * 1e6) 
	//	{
	//		nn.DumpWeights();
	//		//nn.PrintFilters(true);
   //
	//		dump_weight_timer.EndTiming();
	//		dump_weight_timer.BeginTiming();
	//	}
    //
   //
	//	double weight_delta_avg_l1 = vec_dist_L1(nn.weight, w_bak, nn.num_weight) / nn.num_weight;
	//	if(learning_epoch / epoch_reporting_gap == 1) weight_delta_avg_l1 = 0.0;   // delta is 0.0 for the first-time logging
	//	double weight_avg_l1 = vec_avg_L1(nn.weight, nn.num_weight);
     //
	//	printf("==== epoch %lld=====\n", learning_epoch);
	//	nn_timer.Print();
	//	//fprintf(stdout, "%lld \t %lf \t %lf \t avg. l =%lf \t ", epoch, y, neuron[num_neuron-1].f, batch_loss_sum/BATCH_SIZE);
	//	fprintf(stdout, "%lld \t consistency =%.2lf%% \t avg. l =%lf \t d. w =%e \t avg. w =%lf\n", 
	//		learning_epoch, 
	//		action_diff_sum/(double)n_since_last_report*100, 
	//		loss_sum/n_since_last_report, 
	//		weight_delta_avg_l1, 
	//		weight_avg_l1);
	//	printf("===================\n\n");
	//	//getchar();
    //
	//	fprintf(fp_log,	"%lld \t %.2lf%% \t %lf \t %e \t %lf\n", 
	//		learning_epoch, 
	//		action_diff_sum/(double)n_since_last_report*100, 
	//		loss_sum/n_since_last_report, 
	//		weight_delta_avg_l1, 
	//		weight_avg_l1);
	//	fflush(fp_log);
    //
	//	for(int50 i=0; i<nn.num_weight; i++) w_bak[i] = nn.weight[i];
	//	action_diff_sum = 0;
	//	loss_sum = 0;
	//	n_since_last_report = 0;
	//}
     //
	//int50 PrintNNOutput(FILE* fp=stdout)
	//{
	//	int50 argmax = 0;
	//	double max = nn.neuron[nn.offset_output].f;
    //
	//	for(int50 i=0; i<nn.size_output; i++)
	//	{
	//		fprintf(fp, "%.3lf\t", nn.neuron[nn.offset_output+i].f);
	//		if(nn.neuron[nn.offset_output+i].f > max)
	//		{
	//			max = nn.neuron[nn.offset_output+i].f;
	//			argmax = i;
	//		}
	//	}
	//	fprintf(fp, "\n");
	//	return argmax;
	//}
    //
	//void DumpTrainingImgs(double* x, double* a, double r, int50 sample_id, char* foldername =NULL)
	//{
	//	#define DUMPTD_SIZE_ZOOM 4
	//	unsigned char* m_buf = new unsigned char[SIZE_PERCEPT * DUMPTD_SIZE_ZOOM * DUMPTD_SIZE_ZOOM];
	//	char fName[2000];
	//	
	//	int50 nImgs = nn.config[0].H_FC;
	//	int50 size_img = nn.config[0].L_FC;
	//	
	//	if(size_img != nn.config[0].W_FC) throw std::invalid_argument("[RL Error] DumpTrainingImgs() failed because the input of the RL agent is not a square image\n");
	//	if(size_img*size_img*nImgs != SIZE_PERCEPT) throw std::invalid_argument("[RL Error] DumpTrainingImgs() failed due to mismatch in input size\n");
	//	
	//	int50 size_img_zoomed = size_img*DUMPTD_SIZE_ZOOM;
	//	int50 size_img_merged = size_img_zoomed*nImgs;
    //
	//	int50 action = vec_argmax<double>(a, SIZE_ACTION);
    //
	//	for(int50 t=0; t<nImgs; t++)
	//	{	
	//		unsigned char* m = m_buf + (size_img_zoomed)*(size_img_zoomed)*t;
     //
	//		for(int50 p=0; p<(size_img_zoomed)*(size_img_zoomed); p++)
	//		{
	//			int i = (p/(size_img_zoomed))/DUMPTD_SIZE_ZOOM;
	//			int j = (p%(size_img_zoomed))/DUMPTD_SIZE_ZOOM;
	//			
	//			double v = x[(i*size_img+j)*nImgs+t];
	//			m[p] = (int)(v * 256);
	//		}
    //
	//		sprintf(fName, "%s/r%4.3lf_a%lld_d%lld_%lld.bmp", (foldername)?foldername:".", r, action, sample_id, t);
	//		Mat2BMP(m, size_img_zoomed, size_img_zoomed, fName);
	//	}
	//	
	//	// transpose the image vector from a column vector to a row vector
	//	unsigned char m_row_vector[SIZE_PERCEPT * DUMPTD_SIZE_ZOOM*DUMPTD_SIZE_ZOOM];
	//	for(int50 t=0; t<nImgs; t++)
	//	{
	//		for(int50 row=0; row<size_img_zoomed; row++)
	//		{
	//			for(int50 col=0; col<size_img_zoomed; col++)
	//			{
	//				m_row_vector[row*size_img_merged + t*size_img_zoomed + col] = m_buf[t*(size_img_zoomed*size_img_zoomed) + row*size_img_zoomed + col];
	//			}
	//		}
	//	}
	//	memcpy(m_buf, m_row_vector, SIZE_PERCEPT * DUMPTD_SIZE_ZOOM*DUMPTD_SIZE_ZOOM);
    //
	//	sprintf(fName, "%s/r%4.3lf_a%lld_d%lld.bmp", (foldername)?foldername:".", r, action, sample_id);
	//	//Mat2BMP(m_buf, size_img_zoomed*nImgs, size_img_zoomed, fName);
	//	Mat2BMP(m_buf, size_img_zoomed, size_img_zoomed*nImgs, fName);
    //
	//	delete[] m_buf;
	//}
    

};



#endif
