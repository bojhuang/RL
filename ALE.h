

#include <iostream>
#include "ale_interface.hpp"

#include "utils.h"
#include "image.h"
#include "RL.h"

using namespace std;




#define SIZE_IMG		        84
#define SIZE_PERCEPT_ALE        (SIZE_IMG*SIZE_IMG)
#define SIZE_ACTION_SI		    6	        // debug: we use the minimal action set for space invaders here 18
#define FRAME_SKIP		        5	        // it's reported that space invaders needs a frame skip=3
#define SINGLE_LIFE_EPISODE			        // debug: we use single-life mode 
#define STARTING_NOOP	        30			// max number of NOOP steps at the beginning of the round
#define MAX_FRAME_NUM_PER_ROUND	(300*60)    // max time length = 300s per round
#define MAX_STEP_NUM_PER_ROUND	(MAX_FRAME_NUM_PER_ROUND/FRAME_SKIP+1)
//#define ROUND_RECORDING
extern int g_life_cnt;



////
// change list from the original ALE-WIN32 library (all marked with "// hbj")
// 1. added mechanism to pass the life counter out of ALE, on Space Invaders, Breakout
// 2. changed the random seed to the fixed number of 12345
// 3. changed to the max-based frame mixer
// 4. changed the interface of loadROM() so that we can pass more parameters to this routine
// 5. edited SDLKeyboardAgent a little bit so that we can reasonably play ALE games using keyboard
////
class Environment_ALE : public Environment<SIZE_PERCEPT_ALE, SIZE_ACTION_SI>
{
public:
	ALEInterface ale;
	ExportScreen* es;
	ActionVect legal_game_actions;
	
	double current_screen[SIZE_IMG][SIZE_IMG];
    double immediate_reward;
	double game_score;
	int life_cnt;
	
	// logging facilities
	PerfVar timer_fps;
    int50 round;
	int50 step_in_round;	   // count action steps -- which is frame_num x frame_skip + a random number of NOOP actions at the beginning of the round
    PerfVar pv_score;
    PerfVar pv_step_in_round;

public:
    Environment_ALE(const char* filename_rom, bool fDisplay = true)
	    : ale(fDisplay)
	{
        // Load the ROM file
		ale.loadROM(filename_rom, FRAME_SKIP);

		ale.setMaxNumFrames(MAX_FRAME_NUM_PER_ROUND);

		// Get the minimal set of legal actions
		legal_game_actions = ale.getMinimalActionSet();

		es = ale.theOSystem->p_export_screen;

		raw_screen_rgb = new PixelRGB[ale.getScreen().height() * ale.getScreen().width()];
		raw_screen_greylevel = new double[ale.getScreen().height() * ale.getScreen().width()];

        Reset();
        round = 1;

		timer_fps.BeginTiming();	
	}

    virtual ~Environment_ALE()
	{
		delete[] raw_screen_rgb;
		delete[] raw_screen_greylevel;
	}

public:
    virtual void Reset()
	{
		while(1)
		{
			ale.reset_game();

			int50 n_noop = rand() % (STARTING_NOOP+1);
			for(int50 t=0; t<n_noop; t++) ale.act(PLAYER_A_NOOP);
			if(ale.game_over()) 
				printf("[ALE Warning] game over during the starting NOOPs, re-start the game now ...\n");
			else
				break;
		}
		life_cnt = g_life_cnt;
        game_score = 0;
		step_in_round = 0;
		GetScreen(current_screen);
		immediate_reward = 0;
	}

    virtual bool Update(OUT_ double* x, OUT_ double& r, OUT_ bool& fTerminal, IN_ double* a)
    {
        int50 action = vec_argmax<double>(a, SIZE_ACTION_SI);
		assert(action < SIZE_ACTION_SI);
		Action game_action = legal_game_actions[action];
		
        if( GameOver() )
		{
            round++;
            Reset();  
        }
        else
        {
		    // in my laptop (release, x64, frame_skip=5) 
		    // it takes 2ms if no display, 
		    // takes 3ms if we open a minimal window, 
		    // takes 6ms if we use the initial window size
		    immediate_reward = ale.act(game_action);
            
            if( not GameOver() && ale.game_over() )
            {
                // this happens when the game is over while the agent still have lifes,
                // in this case we will go on reducing the life counter to zero without updating ALE;
                // this is to guarantee that under the SINGLE_LIFE_EPISODE mode, the number of episodes per round 
                // will always be equal to the number of agent lives.
                assert(g_life_cnt > 0);
                g_life_cnt --;
		    }
        }
        
        // takes <0.7ms in my laptop (release, x64)		
		GetScreen(current_screen);
		
        memcpy(x, current_screen, SIZE_PERCEPT_ALE*sizeof(double));
        r = immediate_reward;
#ifdef SINGLE_LIFE_EPISODE
		fTerminal = GameOver() || (g_life_cnt < life_cnt);
#else
        fTerminal = GameOver();
#endif
        life_cnt = g_life_cnt;
        game_score += immediate_reward;
		step_in_round++;
        if( GameOver() )
        {
            pv_score.AddRecord(game_score);
            pv_step_in_round.AddRecord(step_in_round);
        }
        timer_fps.EndTiming();
		timer_fps.BeginTiming();
		return true;
    }


    //// screen access routines --------------------------------------------
private:
	struct PixelRGB
	{
		int Red;
		int Green;
		int Blue;
	};
	PixelRGB* raw_screen_rgb;
	double* raw_screen_greylevel;

	// convert the RGB pixel to a grey-level value in [0 , ~0.996]
	inline double RGB2Input(const PixelRGB& pixel)
	{
		// convert RBG to grey-level value (aka. the Y channel),
		// then rescale the grey-level value to [0 , ~0.996]
		return ((pixel.Red * 0.2989) +(pixel.Green * 0.5870) + (pixel.Blue * 0.1140)) / 256;
	}

	// get the current ALE screen, in RGB format
	void GetAleScreen(PixelRGB* buf)
	{
		pixel_t* v = ale.getScreen().getArray();

		for(int p=0; p<ale.getScreen().arraySize(); p++)
		{
			es->get_rgb_from_palette(v[p],buf[p].Red, buf[p].Green, buf[p].Blue);
		}
	}

public:
	//// get the rescaled screen, with pixel value normalized to [0,1]
	//void GetScreen(double current_frame[SIZE_IMG][SIZE_IMG])
	//{
	//	GetAleScreen(raw_screen_rgb);
	//	 
	//	// screen rescaling
	//	// copied from BMP::Rescale() in the EasyBMP.cpp file
	//	int I,J;
	//	double ThetaI,ThetaJ;
	//	int OldHeight = ale.getScreen().height();
	//	int OldWidth = ale.getScreen().width();
	//	int NewHeight = SIZE_IMG;
	//	int NewWidth = SIZE_IMG;
	//	PixelRGB pixel;
	//
	//	#define OldImage(c, r) raw_screen_rgb[(r)*OldWidth+(c)]
	//	//#define RGB2Input(p) ((p).Y() / 256)
	//
	//	for( int j=0; j < NewHeight-1 ; j++ )
	//	{
	//		ThetaJ = (double)(j*(OldHeight-1.0))/(double)(NewHeight-1.0);
	//		J	= (int) floor( ThetaJ );
	//		ThetaJ -= J;  
	// 
	//		for( int i=0; i < NewWidth-1 ; i++ )
	//		{
	//			ThetaI = (double)(i*(OldWidth-1.0))/(double)(NewWidth-1.0);
	//			I = (int) floor( ThetaI );
	//			ThetaI -= I;  
	//  
	//			pixel.Red = (1.0-ThetaI-ThetaJ+ThetaI*ThetaJ)*(OldImage(I,J).Red)
	//							+(ThetaI-ThetaI*ThetaJ)*(OldImage(I+1,J).Red)   
	//							+(ThetaJ-ThetaI*ThetaJ)*(OldImage(I,J+1).Red)   
	//							+(ThetaI*ThetaJ)*(OldImage(I+1,J+1).Red);
	//			pixel.Green = (1.0-ThetaI-ThetaJ+ThetaI*ThetaJ)*OldImage(I,J).Green
	//							+(ThetaI-ThetaI*ThetaJ)*OldImage(I+1,J).Green   
	//							+(ThetaJ-ThetaI*ThetaJ)*OldImage(I,J+1).Green   
	//							+(ThetaI*ThetaJ)*OldImage(I+1,J+1).Green;  
	//			pixel.Blue = (1.0-ThetaI-ThetaJ+ThetaI*ThetaJ)*OldImage(I,J).Blue
	//							+(ThetaI-ThetaI*ThetaJ)*OldImage(I+1,J).Blue   
	//							+(ThetaJ-ThetaI*ThetaJ)*OldImage(I,J+1).Blue   
	//							+(ThetaI*ThetaJ)*OldImage(I+1,J+1).Blue; 
	//
	//			current_frame[j][i] = RGB2Input(pixel);
	//		}
	//
	//		pixel.Red = (1.0-ThetaJ)*(OldImage(OldWidth-1,J).Red)
	//							+ ThetaJ*(OldImage(OldWidth-1,J+1).Red); 
	//		pixel.Green = (1.0-ThetaJ)*(OldImage(OldWidth-1,J).Green)
	//							+ ThetaJ*(OldImage(OldWidth-1,J+1).Green); 
	//		pixel.Blue = (1.0-ThetaJ)*(OldImage(OldWidth-1,J).Blue)
	//							+ ThetaJ*(OldImage(OldWidth-1,J+1).Blue); 
	//
	//		current_frame[j][NewWidth-1] = RGB2Input(pixel);
	//	} 
	//
	//	for( int i=0 ; i < NewWidth-1 ; i++ )
	//	{
	//		ThetaI = (double)(i*(OldWidth-1.0))/(double)(NewWidth-1.0);
	//		I = (int) floor( ThetaI );
	//		ThetaI -= I;  
	//		pixel.Red = (1.0-ThetaI)*(OldImage(I,OldHeight-1).Red)
	//							+ ThetaI*(OldImage(I,OldHeight-1).Red); 
	//		pixel.Green = (1.0-ThetaI)*(OldImage(I,OldHeight-1).Green)
	//							+ ThetaI*(OldImage(I,OldHeight-1).Green); 
	//		pixel.Blue = (1.0-ThetaI)*(OldImage(I,OldHeight-1).Blue)
	//							+ ThetaI*(OldImage(I,OldHeight-1).Blue); 
	//		
	//			current_frame[NewHeight-1][i] = RGB2Input(pixel);
	//	}
	//
	//	current_frame[NewHeight-1][NewWidth-1] = RGB2Input( OldImage(OldWidth-1,OldHeight-1) );
	//		
	//}	

	// get the rescaled screen, with normalized pixel values
	void GetScreen(double current_frame[SIZE_IMG][SIZE_IMG])
	{
		GetAleScreen(raw_screen_rgb);
		
		int raw_screen_size = ale.getScreen().height() * ale.getScreen().width();
		for(int p=0; p<raw_screen_size; p++)
			raw_screen_greylevel[p] = RGB2Input(raw_screen_rgb[p]);
		  
		// screen rescaling
		// copied and modified from BMP::Rescale() in the EasyBMP.cpp file
		int I,J;
		double ThetaI,ThetaJ;
		int OldHeight = ale.getScreen().height();
		int OldWidth = ale.getScreen().width();
		int NewHeight = SIZE_IMG;
		int NewWidth = SIZE_IMG;
		#define OldImage(c, r) raw_screen_greylevel[(r)*OldWidth+(c)]

		for( int j=0; j < NewHeight-1 ; j++ )
		{
			ThetaJ = (double)(j*(OldHeight-1.0))/(double)(NewHeight-1.0);
			J	= (int) floor( ThetaJ );
			ThetaJ -= J;  
  
			for( int i=0; i < NewWidth-1 ; i++ )
			{
				ThetaI = (double)(i*(OldWidth-1.0))/(double)(NewWidth-1.0);
				I = (int) floor( ThetaI );
				ThetaI -= I;  
   
				current_frame[j][i] = (1.0-ThetaI-ThetaJ+ThetaI*ThetaJ)*OldImage(I,J)
											+(ThetaI-ThetaI*ThetaJ)*OldImage(I+1,J)   
											+(ThetaJ-ThetaI*ThetaJ)*OldImage(I,J+1)   
												+(ThetaI*ThetaJ)*OldImage(I+1,J+1);
			}

			current_frame[j][NewWidth-1] = (1.0-ThetaJ)*OldImage(OldWidth-1,J)
											+ ThetaJ*OldImage(OldWidth-1,J+1); 
		} 

		for( int i=0 ; i < NewWidth-1 ; i++ )
		{
			ThetaI = (double)(i*(OldWidth-1.0))/(double)(NewWidth-1.0);
			I = (int) floor( ThetaI );
			ThetaI -= I;  
			current_frame[NewHeight-1][i] = (1.0-ThetaI)*OldImage(I,OldHeight-1)
											+ ThetaI*OldImage(I,OldHeight-1); 
		}

		current_frame[NewHeight-1][NewWidth-1] = OldImage(OldWidth-1,OldHeight-1);
	}

    void PrintScreen(double frame[SIZE_IMG][SIZE_IMG], char* filename)
	{
		#define SIZE_ZOOM 4
		unsigned char m[(SIZE_IMG*SIZE_ZOOM)*(SIZE_IMG*SIZE_ZOOM)];
		int p=0;
		for(int p=0; p<(SIZE_IMG*SIZE_ZOOM)*(SIZE_IMG*SIZE_ZOOM); p++)
		{
			int i = (p/(SIZE_IMG*SIZE_ZOOM))/SIZE_ZOOM;
			int j = (p%(SIZE_IMG*SIZE_ZOOM))/SIZE_ZOOM;
				
			double v = frame[i][j];
			m[p] = (int)(v * 256);
		}

		Mat2BMP(m, SIZE_IMG*SIZE_ZOOM, SIZE_IMG*SIZE_ZOOM, filename);
	}


public:
	double GameFPS(bool reset =false)
	{
		timer_fps.Settle();
		double game_fps = (1e6/timer_fps.mean)*FRAME_SKIP;
		if(reset) timer_fps.Clean();
		return game_fps;
	}

    bool GameOver() 
    {
        return (ale.game_over() && g_life_cnt == 0);
    }

public:
    virtual void Report(FILE* fp=stdout, int detail_level=1, bool reset=true) 
    {
        if(detail_level == 0)
        {
            fprintf(fp, "  ROUND\t" "score/r\t" "#step/r\t" " g-fps\t");
        }
        else if(detail_level == 1)
        {
            pv_score.Settle(); pv_step_in_round.Settle();
            fprintf(fp, "%7lld\t" "%7.1lf\t" "%7.1lf\t" "%6.1lf\t", round, pv_score.mean, pv_step_in_round.mean, GameFPS(reset));
            if(reset) {pv_score.Clean(); pv_step_in_round.Clean();}
        }
        else if(detail_level == 2)
        {  
            GameFPS(reset);
            if(reset) {pv_score.Clean(); pv_step_in_round.Clean();}
        }
    }
};




#define HISTORY_LEN		        4
#define REWARD_CLIP
#define SIZE_PERCEPT_ALEW (SIZE_IMG*SIZE_IMG*HISTORY_LEN)

class Agent_ALEWrapper : public Agent<SIZE_PERCEPT_ALE, SIZE_ACTION_SI>
{
public:
    Agent<SIZE_PERCEPT_ALEW, SIZE_ACTION_SI>* pAgent;
    Queue<double[SIZE_IMG][SIZE_IMG]> screen_history;
    double percept_buf[SIZE_IMG][SIZE_IMG][HISTORY_LEN];

public:
    Agent_ALEWrapper(Agent<SIZE_PERCEPT_ALEW, SIZE_ACTION_SI>* pAgent_rhs) 
        : pAgent(pAgent_rhs), screen_history(HISTORY_LEN-1)
    {
        double null_frame[SIZE_IMG][SIZE_IMG] = {{0}};
        for(int50 t=0; t<HISTORY_LEN-1; t++) screen_history.push_back(null_frame, true);
    }
    virtual ~Agent_ALEWrapper() {}

    double RewardClip(double r_raw)
	{
#ifdef REWARD_CLIP
        #define R_MAX  1.0
        #define R_MIN -1.0
		return (r_raw > R_MAX) ? (R_MAX) : (r_raw < R_MIN) ? (R_MIN) : r_raw;
#else
		return r_raw;
#endif
    }

    // construct the percept signal from 'screen_history' and the current screen 'x', and fill it in the percept buffer
    double* GenPercept(IN_ double* x)
    {
        for(int50 i=0; i<SIZE_IMG; i++)
        for(int50 j=0; j<SIZE_IMG; j++)
        {
            for(int t=0; t<HISTORY_LEN-1; t++)
            {
                percept_buf[i][j][t] = screen_history[t][i][j];
            }
            percept_buf[i][j][HISTORY_LEN-1] = x[i*SIZE_IMG+j];
        }
        return (double*)percept_buf;
    }

    // print all frames in the percept (note that the format of percept is different from the screen_history)
	void PrintPerceptBuffer(char* filename_prefix =NULL) const
	{
		for(int t=0; t<HISTORY_LEN; t++)
		{
			char fName[2000];
			sprintf(fName, "%s_%lld.bmp", (filename_prefix) ? filename_prefix : "", t);

			#define SIZE_ZOOM 4
			unsigned char m[(SIZE_IMG*SIZE_ZOOM)*(SIZE_IMG*SIZE_ZOOM)];
			int p=0;
			for(int p=0; p<(SIZE_IMG*SIZE_ZOOM)*(SIZE_IMG*SIZE_ZOOM); p++)
			{
				int i = (p/(SIZE_IMG*SIZE_ZOOM))/SIZE_ZOOM;
				int j = (p%(SIZE_IMG*SIZE_ZOOM))/SIZE_ZOOM;
				
				double v = percept_buf[i][j][t];
				m[p] = (int)(v * 256);
			}

			Mat2BMP(m, SIZE_IMG*SIZE_ZOOM, SIZE_IMG*SIZE_ZOOM, fName);
		}
		return;
	}

public:
    virtual bool TakeAction(OUT_ double* a, IN_ double* x, IN_ double r, IN_ bool fTerminal=false)
    {
        pAgent->TakeAction(a, GenPercept(x), RewardClip(r), fTerminal);

        //debug 
        //PrintPerceptBuffer("percept");

        if(HISTORY_LEN>1)
        {
            double x_buf[SIZE_IMG][SIZE_IMG];
            memcpy(x_buf, x, SIZE_IMG*SIZE_IMG*sizeof(double));
            screen_history.push_back(x_buf, true);
        }

        //// clear screen_history between episodes
        //if(fTerminal == true)
        //{
        //    double null_frame[SIZE_IMG][SIZE_IMG] = {{0}};
        //    for(int t=0; t<HISTORY_LEN-1; t++) screen_history.push_back(null_frame, true);
        //}

        return true;
    }

    virtual bool Act(OUT_ double* a, IN_ double* x)
    {
        return pAgent->Act(a, GenPercept(x));
    }

	virtual bool Learn(IN_ double* x, IN_ double* a, IN_ double r)
    {
        return pAgent->Learn(GenPercept(x), a, RewardClip(r));
    }

    virtual void Print(FILE* fp=stdout) 
    {
        pAgent->Print(fp);
    }

    virtual void Report(FILE* fp=stdout, int50 detail_level=1, bool reset=true) 
    {
        pAgent->Report(fp, detail_level, reset);
    }


};


