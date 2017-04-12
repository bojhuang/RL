#ifndef __RL_H__
#define __RL_H__


#include <deque>
#include "utils.h"
#include "PerfVar.h"


template<int50 SIZE_PERCEPT, int50 SIZE_ACTION>
class Environment
{
public:
	virtual ~Environment(){}

    virtual void Reset() =0;
	virtual bool Update(OUT_ double x[SIZE_PERCEPT], OUT_ double& r, OUT_ bool& fTerminal, IN_ double a[SIZE_ACTION]) =0;
    
    virtual void Print(FILE* fp=stdout) {}
    virtual void Report(FILE* fp=stdout, int detail_level=1, bool reset=true) {}
};


template<int50 SIZE_PERCEPT, int50 SIZE_ACTION>
class Agent
{
protected:
	class MemItem
	{
	public:
		double x[SIZE_PERCEPT];
		double a[SIZE_ACTION];
		double r_lt;
		int50 step; // for logging and debugging
		double r_st;  // for logging and debugging

		MemItem(IN_ double* xx =NULL, IN_ double* aa =NULL, IN_ double rr_lt =0.0, IN_ int50 ss =0, IN_ double rr_st =0)
		{
			if(xx!=NULL) for(int50 i=0; i<SIZE_PERCEPT; i++) x[i] = xx[i];
			if(aa!=NULL) for(int50 i=0; i<SIZE_ACTION; i++) a[i] = aa[i];
			r_lt = rr_lt;
			step = ss;
			r_st = rr_st;
		}

        void CopyToBuffer (OUT_ double* buffer_x, OUT_ double * buffer_a, OUT_ double& buffer_r, OUT_ int50& buffer_step, OUT_ double& buffer_rst) const
        {
            for(int50 i=0; i<SIZE_PERCEPT; i++) buffer_x[i] = x[i];
            for(int50 i=0; i<SIZE_ACTION; i++) buffer_a[i] = a[i];
            buffer_r = r_lt;
            buffer_step = step;
            buffer_rst = r_st;
        }
	};
	std::deque<MemItem> stm;

public:
	virtual ~Agent(){}

	virtual bool Act(OUT_ double* a, IN_ double* x) =0;

	virtual bool Learn(IN_ double* x, IN_ double* a, IN_ double r) =0;

	virtual bool TakeAction(OUT_ double* a, IN_ double* x, IN_ double r, IN_ bool fTerminal=false)
	{
		if(false == Act(a, x)) return false;
		
		if(stm.empty() == false)
		{
			// update short-term memory
			stm.back().r_lt = r;
		
			// update long-term memory
			if(Learn(stm.front().x, stm.front().a, stm.front().r_lt) == false) return false;
			stm.pop_front();
		}

		// add the latest experience
		stm.emplace_back(x, a, 0.0);
		//stm.push_back(STMemItem(x, a, 0.0));
		
		
		return true;
	}

    virtual void Print(FILE* fp=stdout) {} 
    virtual void Report(FILE* fp=stdout, int50 detail_level=1, bool reset=true) {}
};



template<int50 SIZE_PERCEPT, int50 SIZE_ACTION>
void RL_Experiment(
    Environment<SIZE_PERCEPT, SIZE_ACTION>& env, 
    Agent<SIZE_PERCEPT, SIZE_ACTION>& agent, 
    int50 nEpisode =100, int50 episode_reporting_cycle =1,
    int50 nStep =-1, int50 step_reporting_cycle =-1
    )
{
    // initialize the experiment
    double x[SIZE_PERCEPT] = {0};
    double r = 0;
    bool fTerminal = false;
    double a[SIZE_ACTION] = {0};
    FILE* fp_log = fopen("log.txt", "w");
    FileSynchronizer log("log.txt");
    FILE* fp_env = fopen("log_env.txt", "w");
    FILE* fp_agent = fopen("log_agent.txt", "w");

    int50 step = 1;
    int50 episode = 1;
    int50 step_cnt = 0;
    double r_sum = 0;
    PerfVar pv_step_cnt("#step");
    PerfVar pv_r_sum("r_ep."); // per-episode reward averaged over all episodes in the reporting cycle
    PerfVar pv_r_avg("r_avg"); // per-step reward averaged directly over the whole reporting cycle
    PerfVar pv_fps("FPS");
    PerfVar pv_timer_env("Env.");
    PerfVar pv_timer_agent("Agent");
    PerfVar wall_clock;

    env.Update(x, r, fTerminal, a);
    wall_clock.BeginTiming();
    
    while(true)
    {
        if(nStep > 0 && step > nStep) break;
        if(nEpisode > 0 && episode > nEpisode) break;
        pv_fps.BeginTiming();

        pv_timer_agent.BeginTiming();
        agent.TakeAction(a, x, r, fTerminal);
        pv_timer_agent.EndTiming();

        pv_timer_env.BeginTiming();
        env.Update(x, r, fTerminal, a);
        pv_timer_env.EndTiming();

        // logging
        step_cnt++;
        r_sum += r;
        if(step_reporting_cycle > 0 && step % step_reporting_cycle == 0)
        {
            fprintf(fp_log, "\nReport [Step %lld] -------\n", step);
            env.Report(fp_log);
            fprintf(fp_log, "\n");
            agent.Report(fp_log);
            fprintf(fp_log, "\n");
        }
        if(fTerminal == true)
        {
            pv_step_cnt.AddRecord(step_cnt);
            pv_r_sum.AddRecord(r_sum);

            if(episode_reporting_cycle > 0 && episode % episode_reporting_cycle == 0)
            {  
                static int50 report_id = 0;
                
                int50 print_title_cycle = 45;
                if(fp_log == stdout || fp_log == stderr) print_title_cycle = 1;

                if(report_id % print_title_cycle == 0)
                {
                    fprintf(fp_log, "EXP_STEP " "\t episode\t" " #step \t" " r_episode \t" "   fps\t" "  T_env\t" "T_agent\t");
                    env.Report(fp_log, 0);
                    agent.Report(fp_log, 0);
                    fprintf(fp_log, "#report\t""clock\t""\n");
                }
                
                fprintf(fp_log, "%7lld\t", step);
                if(episode_reporting_cycle == 1) fprintf(fp_log, "\t%7lld \t", episode);
                else fprintf(fp_log, "%7lld - %7lld\t", episode-episode_reporting_cycle+1, episode);
                
                pv_step_cnt.Settle(); fprintf(fp_log, "%7.1lf\t", pv_step_cnt.mean); pv_step_cnt.Clean();
                
                pv_r_sum.Settle(); fprintf(fp_log, "%10.3lf\t", pv_r_sum.mean); pv_r_sum.Clean();    

                pv_fps.Settle(); fprintf(fp_log, "%6.1lf\t", 1000000 / pv_fps.mean); pv_fps.Clean();

                pv_timer_env.Settle(); fprintf(fp_log, "%7.1lf\t", pv_timer_env.mean /1000); pv_timer_env.Clean();

                pv_timer_agent.Settle(); fprintf(fp_log, "%7.1lf\t", pv_timer_agent.mean / 1000); pv_timer_agent.Clean();

                env.Report(fp_log, 1, false); 
                agent.Report(fp_log, 1, false);
                fprintf(fp_log, "%7lld\t""%.2lfh""\n", report_id, wall_clock.CheckTime()/3600/1000000);
                if(fp_log == stdout || fp_log == stderr) fprintf(fp_log, "\n");
                fflush(fp_log);

                fprintf(fp_env, "\nEpisode Report #%lld -------\n", report_id);
                env.Report(fp_env, 2, true);  
                fflush(fp_env);

                fprintf(fp_agent, "\nEpisode Report #%lld -------\n", report_id);
                agent.Report(fp_agent, 2, true);
                fflush(fp_agent);

                report_id++;

                if( not (fp_log == stdout || fp_log == stderr) ) log.Sync();
            }

            episode ++;
            step_cnt = 0;
            r_sum = 0;
        }
        pv_r_avg.AddRecord(r);
        step ++;

        pv_fps.EndTiming();
    }
    
    return;
}


#endif

