
#include "utils.h"
#include "RL.h"
#include "Agent_DQN.h"

//select game type
#define MNIST
//#define TTT
//#define ALE


#if defined(MNIST)
#include "mnist.h"
// mnist_main

void mnist_main()
{
    // create the agent
    static const int50 batch_size = 32;
    int50 num_parallel_learner = 2;
    char* filename_weight = NULL;
	std::vector<DCNN_CONFIG> config;

	// Foraging 10x10
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	5,	0,  1,  0, 10, 10));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		10,	3,  1, -1, 10, 10));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		20, 3,  1,  0,  8,  8));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		40, 3,  1,  0,  6,  6));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		80, 3,  1,  0,  4,  4)); 
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,	9,	4,  1,  0,  1,  1));

	//// aDQN (4-way input)
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	4,	0,	1,	0,	84,	84));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		16,	8,	4,	0,	20,	20));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		32,	4,	2,	0,	9,	9));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		256,9,	1,	0,	1,	1));
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		SIZE_ACTION,	1,	1,	0,	1,	1));

	//// DQN (4-way input)
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	4,	0,	1,	0,	84,	84));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		32,	8,	4,	0,	20,	20));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		64,	4,	2,	0,	9,	9));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		64,	3,	1,	0,	7,	7));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		512,7,	1,	0,	1,	1));
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		18,	1,	1,	0,	1,	1));

	// MNIST
	config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	1,	0,  0,  0, 28, 28));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		8,	5,  1, -2, 28, 28));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		16, 3,  2, -1, 14, 14));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		32, 3,  2, -1,  7,  7));
	config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		10, 7,  1,  0,  1,  1));

    Agent_DQN<SIZE_IMG, SIZE_LABEL> agent(config, filename_weight, batch_size, num_parallel_learner, 400, 60000, 0.1, 0.0);


    // create the envionment
    Environment_Mnist env(10);


    // do the experiment
    int50 nStep = -1; //120000;
    int50 step_reporting_cycle = -1; //100 * UPDATE_INTERVAL;
    int50 nEpisode = 120000;
    int50 episode_reporting_cycle = 100 * UPDATE_INTERVAL;

    RL_Experiment(env, agent, nEpisode, episode_reporting_cycle);

    return;
}

#elif defined(ALE)

#include "ALE.h"

// ale_main

void ale_main()
{
    // create the agent
    static const int50 batch_size = 32;
    int50 num_parallel_learner = 16;
    char* filename_weight = NULL;
	std::vector<DCNN_CONFIG> config;

	// Foraging 10x10
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	5,	0,  1,  0, 10, 10));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		10,	3,  1, -1, 10, 10));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		20, 3,  1,  0,  8,  8));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		40, 3,  1,  0,  6,  6));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		80, 3,  1,  0,  4,  4)); 
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,	9,	4,  1,  0,  1,  1));

	// aDQN (4-way input)
	config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	HISTORY_LEN,	0,	1,	0,	84,	84));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		16,	8,	4,	0,	20,	20));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		32,	4,	2,	0,	9,	9));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		256,9,	1,	0,	1,	1));
	config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		SIZE_ACTION_SI,	1,	1,	0,	1,	1));

	//// DQN (4-way input)
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	4,	0,	1,	0,	84,	84));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		32,	8,	4,	0,	20,	20));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		64,	4,	2,	0,	9,	9));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		64,	3,	1,	0,	7,	7));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		512,7,	1,	0,	1,	1));
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		18,	1,	1,	0,	1,	1));

	//// MNIST
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	1,	0,  0,  0, 28, 28));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		8,	5,  1, -2, 28, 28));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		16, 3,  2, -1, 14, 14));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		32, 3,  2, -1,  7,  7));
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		10, 7,  1,  0,  1,  1));

    //Agent_DQN<SIZE_PERCEPT_ALE, SIZE_ACTION_SI> agent_dqn(config, filename_weight, batch_size, num_parallel_learner, 5000, 100000, 1.0, 0.1);
    Agent_DQN<SIZE_PERCEPT_ALEW, SIZE_ACTION_SI> agent_dqn(config, filename_weight, batch_size, num_parallel_learner, 5000, 100000, 1.0, 0.1);


    // create the envionment
    Environment_ALE env("roms/space_invaders.bin", true);

    // create the agent wrapper
    Agent_ALEWrapper agent(&agent_dqn);

    // do the experiment
    int50 nStep = -1; //120000;
    int50 step_reporting_cycle = -1; //100 * UPDATE_INTERVAL;
    int50 nEpisode = 50000;
#ifdef SINGLE_LIFE_EPISODE
    int50 episode_reporting_cycle = 3;
#else
    int50 episode_reporting_cycle = 1;
#endif
    RL_Experiment(env, agent, nEpisode, episode_reporting_cycle);

    return;
}

#elif defined(TTT)

#include "tic_tac_toe.h"

// ttt_main

void ttt_main()
{
    static const int50 BOARD_SIZE = 3;
    static const int50 CHAIN_SIZE = 3;

    // create the agent
    static const int50 batch_size = 32;
    int50 num_parallel_learner = 2;
    char* filename_weight = NULL;
	std::vector<DCNN_CONFIG> config;

	// Foraging 10x10
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	5,	0,  1,  0, 10, 10));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		10,	3,  1, -1, 10, 10));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		20, 3,  1,  0,  8,  8));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		40, 3,  1,  0,  6,  6));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		80, 3,  1,  0,  4,  4)); 
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,	9,	4,  1,  0,  1,  1));

	//// aDQN (4-way input)
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	4,	0,	1,	0,	84,	84));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		16,	8,	4,	0,	20,	20));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		32,	4,	2,	0,	9,	9));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		256,9,	1,	0,	1,	1));
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		SIZE_ACTION,	1,	1,	0,	1,	1));

	//// DQN (4-way input)
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	4,	0,	1,	0,	84,	84));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		32,	8,	4,	0,	20,	20));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		64,	4,	2,	0,	9,	9));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		64,	3,	1,	0,	7,	7));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		512,7,	1,	0,	1,	1));
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		18,	1,	1,	0,	1,	1));

	//// MNIST
	//config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	1,	0,  0,  0, 28, 28));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		8,	5,  1, -2, 28, 28));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		16, 3,  2, -1, 14, 14));
	//config.push_back(DCNN_CONFIG(NTYPE_RELU,		32, 3,  2, -1,  7,  7));
	//config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		10, 7,  1,  0,  1,  1));

    // Tic-Tac-Toe (3,3)
	config.push_back(DCNN_CONFIG(NTYPE_CONSTANT,	3,	0,  0,  0, 3, 3));
	config.push_back(DCNN_CONFIG(NTYPE_RELU,		100,3,  1,  0, 1, 1));
	config.push_back(DCNN_CONFIG(NTYPE_LINEAR,		9,  1,  1,  0, 1, 1));


    Agent_DQN<SIZE_PERCEPT_T3W, SIZE_ACTION_T3W> agent_core(config, filename_weight, batch_size, num_parallel_learner, 5000, 100000, 1.0, 0.1);
    Agent_TTTWrapper<BOARD_SIZE, CHAIN_SIZE> agent(&agent_core, false);


    // create the envionment
    //AgentKB_TTT<BOARD_SIZE, CHAIN_SIZE> agent_oppo;
    //Agent_DQN<SIZE_PERCEPT_T3W, SIZE_ACTION_T3W> agent_oppo_core(config, filename_weight, batch_size, num_parallel_learner, 5000, 100000, 1.0, 0.1);
    //Agent_TTTWrapper<BOARD_SIZE, CHAIN_SIZE> agent_oppo(&agent_oppo_core, false);
    Agent_TTTMinimax<BOARD_SIZE,CHAIN_SIZE> agent_oppo;
    Environment_TTT<BOARD_SIZE, CHAIN_SIZE> env(&agent_oppo, EMPTY, false); 

    // do the experiment
    int50 nStep = -1; //120000;
    int50 step_reporting_cycle = -1; //100 * UPDATE_INTERVAL;
    int50 nEpisode = 100000;
    int50 episode_reporting_cycle = 100;

    RL_Experiment(env, agent, nEpisode, episode_reporting_cycle);
    agent_core.nn.DumpWeights();


    Agent_DQN<SIZE_PERCEPT_T3W, SIZE_ACTION_T3W> agent_test_core(config, "weight.bin", 0, 0, 100000, 100001, 0.0, 0.0);
    Agent_TTTWrapper<BOARD_SIZE, CHAIN_SIZE> agent_test(&agent_test_core, false);

    printf("\n\n======================\n");
    printf("Testing Stage\n");
    printf("======================\n\n");
    RL_Experiment(env, agent_test, 1000, 1);

    ttt_test<BOARD_SIZE, CHAIN_SIZE>(&agent_test, EMPTY);

    return;
}

#endif // test type

int main(int argc, char *argv[])
{

#if defined(MNIST)
    mnist_main();
#elif defined(ALE)
    ale_main();
#elif defined(TTT)
    ttt_main();

    Agent_TTTMinimax<3,3> agent_minimax;
    ttt_test<3,3>(&agent_minimax, BLACK);

#endif
    return 0;
}



//
//void TestingScript(const char* filename_rom, const double epsilon_min, const int50 num_round)
//{
//	const char* filename_score = "testing_score.tsv";
//	const char* filename_step = "testing_step.tsv";
//	const char* filename_details = "testing_details.txt";
//	RemoveFile(filename_score);
//	RemoveFile(filename_step);
//	FILE* fp_details = fopen(filename_details, "w");
//
//	for(int50 epoch=1; epoch<=200; epoch++)
//	{
//		char filename_weight[2000];
//		sprintf(filename_weight, "../weight_%dk.bin", epoch*250);
//		/*while(ExistFile(filename_weight) == false)
//		{
//			Sleep(300000);
//		}*/
//		if(ExistFile(filename_weight) == false) continue;
//
//		EnvironmentALE env(filename_rom, true);
//		Agent_DCNN<SIZE_PERCEPT, SIZE_ACTION> agent(1, 10, 20, 0, epsilon_min, filename_weight);
//
//		srand(epoch);
//		volatile static int50 seed_sum = 0;
//		for(int i=0; i<100; i++) seed_sum += rand();
//
//		double x[SIZE_PERCEPT];
//		double a[SIZE_ACTION];
//		double r=0;
//
//		PerfVar pv_score(filename_weight);
//		PerfVar pv_step(filename_weight);
//		printf(	"\ntesting %s\n", filename_weight);
//		fprintf(fp_details,
//				"\ntesting %s\n", filename_weight);
//
//		// results of the first round are throwed out
//		for (int50 round=0; round<=num_round; round++) 
//		{
//			while (1) 
//			{
//				env.GenPercept(x);
//				agent.TakeAction(r, x, a, env.fTerminal);
//				env.Update(a);
//				env.GenReward(r);
//				if(env.ale.game_over()) break;
//			}
//
//			if(round > 0)
//			{
//				agent.actor_activeness.Settle();
//				double activeness = agent.actor_activeness.mean;
//				agent.actor_activeness.Clean(); 
//
//				pv_score.AddRecord(env.game_score);
//				pv_step.AddRecord(env.step_in_round);
//				printf( "round %lld: \t score = %.0lf \t #step = %lld \t activeness = %.3lf\n", round, env.game_score, env.step_in_round, activeness);
//				fprintf(fp_details,
//						"round %lld: \t score = %.0lf \t #step = %lld \t activeness = %.3lf\n", round, env.game_score, env.step_in_round, activeness);
//			}
//
//			env.ResetGame();
//		} //.~ round
//
//		pv_score.Dump(filename_score);
//		pv_step.Dump(filename_step);
//		printf(	"\t \t avg. score = %.1lf \t avg. #step = %.1lf\n\n", pv_score.mean, pv_step.mean);
//		fprintf(fp_details,
//				"\t \t avg. score = %.1lf \t avg. #step = %.1lf\n\n", pv_score.mean, pv_step.mean);
//		fflush(fp_details);
//	} //.~ epoch
//}
//
//
//int main(int argc, char** argv) 
//{	
//	if(argc == 5 && strcmp(argv[1], "test")==0)
//	{
//		char* filename_rom = argv[2];
//		double epsilon_min;
//		int50 num_round;
//		sscanf(argv[3], "%lf", &epsilon_min);
//		sscanf(argv[4], "%lld", &num_round);
//		TestingScript(filename_rom, epsilon_min, num_round);
//		return 0;
//	}
//	//else
//	//{
//	//	printf("[usage] %s test rom_file epsilon_min num_round\n", argv[0]);
//	//	return -1;
//	//}
//
//	/*int thread_num = 1;
//	if (argc > 1) sscanf(argv[1], "%d", &thread_num);
//
//	ThroughputTest(thread_num);
//	return 0;*/
//
//
//    if (argc < 2 || argc > 9) 
//	{
//        std::cerr << "Usage: " << argv[0] << " rom_file" << " [display_mode?]" << " [num_worker]" << " [rpm_min_size]" << " [rpm_max_size]" << " [batch_size]" << " [epsilon]" << " [weight_file]" << std::endl;
//		return 1;
//    }
//
//	char* filename_rom = argv[1];
//	bool display_screen = true; // by default, display screen 
//	if(argc>=3)
//	{
//		if(strcmp(argv[2], "true")==0 || strcmp(argv[2], "yes")==0 || strcmp(argv[2], "y")==0 || strcmp(argv[2], "Y")==0)
//			display_screen = true;
//		else if (strcmp(argv[2], "false")==0 || strcmp(argv[2], "no")==0 || strcmp(argv[2], "n")==0 || strcmp(argv[2], "N")==0)
//			display_screen = false;
//		else
//		{
//			printf("[display_mode?] = true|false, yes|no, y|n, Y|N\n");
//			return -1;
//		}
//	}
//
//	int50 num_worker;
//	int50 rpm_min_size;
//	int50 rpm_max_size;
//	int50 batch_size;
//	double epsilon;
//	char* filename_weight;
//	if(argc >= 4)
//	{
//		sscanf(argv[3], "%lld", &num_worker);
//		if(num_worker < 1) 
//		{
//			printf("[error] num_worker must be at least 1\n");
//			return -1;
//		}
//	}
//	else
//		num_worker = DEFAULT_NUM_WORKER;
//
//	if(argc >= 5)
//	{
//		sscanf(argv[4], "%lld", &rpm_min_size);
//		if(rpm_min_size < 10) 
//		{
//			printf("[error] rpm_min_size must be at least 10\n");
//			return -1;
//		}
//	}
//	else
//		rpm_min_size = DEFAULT_RPM_MIN_SIZE;
//
//	if(argc >= 6)
//	{
//		sscanf(argv[5], "%lld", &rpm_max_size);
//		if(rpm_max_size < rpm_min_size) 
//		{
//			printf("[error] rpm_max_size must be no smaller than rpm_min_size\n");
//			return -1;
//		}
//	}
//	else
//		rpm_max_size = DEFAULT_RPM_MAX_SIZE;
//
//	if(argc >= 7)
//	{
//		sscanf(argv[6], "%lld", &batch_size);
//		if(batch_size < 0) 
//		{
//			printf("[error] batch_size cannot be a negative number\n");
//			return -1;
//		}
//	}
//	else
//		batch_size = DEFAULT_BATCH_SIZE;
//
//	if(argc >= 8)
//	{
//		sscanf(argv[7], "%lf", &epsilon);
//		if(epsilon < 0.0 || epsilon >1.0) 
//		{
//			printf("[error] the epsilon for epsilon-greedy must be in [0,1]\n");
//			return -1;
//		}
//	}
//	else
//		epsilon = DEFAULT_EPSILON_MIN;
//
//	
//	if(argc >= 9)
//	{
//		filename_weight = argv[8];
//	}
//	else
//		filename_weight = NULL;
//
//	printf("display_mode = %s\n", (display_screen) ? "true" : "false");
//	printf("num_worker = %lld\n", num_worker);
//	printf("rpm_min_size = %lld\n", rpm_min_size);
//	printf("rpm_max_size = %lld\n", rpm_max_size);
//	printf("batch_size = %lld\n", batch_size);
//	printf("epsilon = %lf\n", epsilon);
//	printf("weight file = %s\n", (filename_weight)? filename_weight : "NULL");
//
//
//	/*//// debug
//	//int50 epoch = 1;
//	//int50 learning_round = 0;
//	//int50 action_round = 0;
//	//int50 rpm_size = 0;
//	//bool mt_progress = false;
//	//int50 rng_cnt = 0;
//
//	//std::uniform_real_distribution<double> u_0_1(0.0, 1.0);
//	//u_0_1(agent.rng);
//	//u_0_1(agent.rng);
//	//action_round ++;
//	//
//	//for (int50 episode_round=1; episode_round<100000; episode_round++)
//	//{
//	//	 	rpm_size++;
//
//	//		double epsilon = 1.0 - (1.0-epsilon_min)/(double)annealing_length * (double)(action_round - annealing_start);
//	//	epsilon = (epsilon > 1.0) ? 1.0 : (epsilon < epsilon_min) ? epsilon_min : epsilon;
//	//	if(u_0_1(rng) < epsilon)
//
//	//	// actor thread
//	//	if(rpm_size < 50000)
//	//	{
//	//		std::uniform_real_distribution<double> u_0_1(0.0, 1.0);
//	//		u_0_1(agent.rng);		 rng_cnt++;
//	//		u_0_1(agent.rng);		 rng_cnt++;	
//	//	}
//	//	else
//	//	{
//	//		std::uniform_int_distribution<int50> rpm_sampler(0, (action_round%4 == 0)?(rpm_size-2):(rpm_size-1));
//
//	//		int50 data_id1 = rpm_sampler(agent.rng);   rng_cnt++;
//	//		printf("%lld \t %lld\n", rng_cnt, data_id1);
//	//		int50 data_id2 = rpm_sampler(agent.rng);   rng_cnt++;
//	//		printf("%lld \t %lld\n", rng_cnt, data_id2);
//	//	}
//	//	
//	//	action_round ++;
//
//	//	 
//
//	//	if(rpm_size >= 50000 && action_round%4 == 0)
//	//	{
//	//		if(mt_progress == true)
//	//		{
//	//			epoch++;
//	//			learning_round += 32;
//	//			mt_progress = false;
//	//		}
//	//		mt_progress = true;
//
//	//		printf("epoch %lld\n", epoch);
//	//		for(int50 t=0; t<32; t++)
//	//		{
//	//			std::uniform_int_distribution<int50> rpm_sampler(0, rpm_size-1);
//	//			int50 data_id = rpm_sampler(agent.rng);	  rng_cnt++;
//	//			printf("%lld \t %lld\n", rng_cnt, data_id);
//	//		}
//	//		printf("\n");
//	//	}
//	//}	 */
//
//	EnvironmentALE env(filename_rom, display_screen);
//	Agent_DCNN<SIZE_PERCEPT, SIZE_ACTION> agent(num_worker, rpm_min_size, rpm_max_size, batch_size, epsilon, filename_weight);
//	srand(54321);
//
//	for(int i=0; i<env.legal_game_actions.size(); i++)
//	{
//		printf("action %d = game action %d\n", i, env.legal_game_actions[i]);
//	}
//
//
//	double x[SIZE_PERCEPT];
//	double a[SIZE_ACTION];
//	double r=0;
//
//	FILE* fp_timer=fopen("timer.txt","w");
//	FILE* fp_score=fopen("score.txt", "w");
//	fprintf(fp_score, "steps \t round\t #action \t activeness \t score \t game value \t game value(est.)\n");
//	PerfVar pv_env("Env.   ");
//	PerfVar pv_agent("Agent  ");
//	int50 steps = 0;
//
//
//    for (int50 round=0; round<=100000; round++) 
//	{
//		double game_value_estimated = 0; 
//		double game_value_real = 0;
//
//        while (1) 
//		{
//			pv_env.BeginTiming();
//			env.GenPercept(x);
//			pv_env.SuspendTiming();
//
//			// debug
//			//rand();
//
//			// logging
//			if(round % round_report_gap == 0)
//			{
//				// record the estmiated game-value at the beginning
//				if(env.step_in_round == 0)
//				{
//					agent.nn.forward(x);
//					for(int50 i=0; i<SIZE_ACTION; i++) 
//					{
//						a[i] = agent.nn.neuron[agent.nn.offset_output+i].f;
//					}	
//					game_value_estimated = vec_max<double>(a, SIZE_ACTION);
//				}
//			}
//
//			pv_agent.BeginTiming();
//			agent.TakeAction(r, x, a, env.fTerminal);
//			pv_agent.EndTiming();
//
//
//			pv_env.ResumeTiming();
//			env.Update(a);
//			env.GenReward(r);
//			game_value_real += r;
//
//			steps++;
//			if(steps%250000 == 0)
//			{
//				char filename[200];
//				sprintf(filename, "weight_%lldk.bin", steps/1000);
//				agent.nn.DumpWeights(filename);
//			}
//			pv_env.EndTiming();
//
//			if(env.ale.game_over()) break;
//        }
//		
//		// logging
//		if(round % round_report_gap == 0)
//		{
//			//env.timer_frame.SuspendTiming();
//			double game_fps = env.GameFPS();
//			agent.actor_activeness.Settle();
//			double activeness = agent.actor_activeness.mean;
//			agent.actor_activeness.Clean(); 
//
//			printf("[Info] game round %d ...............\n", round);
//			printf("total steps = %lld\n", env.step);
//			printf("#action= %lld\n", env.step_in_round);
//			printf("score= %lf\n", env.game_score);
//			printf("game value (real|estimated)= %lf | %lf\n", game_value_real, game_value_estimated);
//			printf("FPS (game|action) = %.1lf | %.1lf\n", game_fps, game_fps/FRAME_SKIP);
//			printf("actor_activeness= %.3lf\n", activeness);
//			pv_env.Print();
//			pv_agent.Print();
//			agent.actor_timer.Print();
//			agent.learner_timer.Print();
//			agent.sgd_timer.Print();
//			agent.mt_timer.Print();
//
//			fprintf(fp_score, "%lld \t %lld \t %lld \t %lf \t %lf \t %lf \t %lf\n", 
//				env.step,
//				round, 
//				env.step_in_round, 
//				activeness,
//				env.game_score, 
//				game_value_real, 
//				game_value_estimated);
//			fflush(fp_score);
//			
//			fprintf(fp_timer, "\nround %lld\n", round);
//			fprintf(fp_timer, "game fps = %.1lf \t action fps = %.1lf\n", game_fps, game_fps/FRAME_SKIP);
//			pv_env.Print(fp_timer);		pv_env.Clean();
//			pv_agent.Print(fp_timer);	pv_agent.Clean();
//			agent.actor_timer.Print(fp_timer); agent.actor_timer.Clean();
//			agent.learner_timer.Print(fp_timer); agent.learner_timer.Clean();
//			agent.sgd_timer.Print(fp_timer); agent.sgd_timer.Clean();
//			agent.mt_timer.Print(fp_timer); agent.mt_timer.Clean();
//			fflush(fp_timer);
//
//			// screen snapshots
//			if( env.screen_history.size() != env.step_in_round ||
//				env.game_action_history.size() != env.step_in_round ||
//				env.reward_history.size() != env.step_in_round ||
//				env.score_history.size() != env.step_in_round)
//			{
//				printf("[Game Error] inconsistent game history length\n");
//				getchar();
//			}
//
//#ifdef ROUND_RECORDING
//			system("del /Q screenshot\\*.bmp");
//			for(int t=0; t<env.step_in_round; t++)
//			{
//				char filename[2000];
//				sprintf(filename, "screenshot/%d_a%d_r%.0lf_s%.0lf.bmp", t, env.game_action_history[t], env.reward_history[t], env.score_history[t]);
//				env.PrintScreen(env.screen_history[t], filename);
//			}
//#endif
//
//			printf("\n\n");
//			//env.timer_frame.ResumeTiming();
//		}
//
//		env.ResetGame();
//
//    } //.~ round
//
//	return 0;
//};
//
//
