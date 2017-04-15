#ifndef __TIC_TAC_TOE_H__
#define __TIC_TAC_TOE_H__


#include "RL.h"



enum TTTCell 
{
    EMPTY = 0,
    WHITE = 1,
    BLACK = 2
};
typedef TTTCell TTTPlayer;


template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
class GameTTT
{
public:
    TTTCell board[BOARD_SIZE][BOARD_SIZE];
    TTTPlayer player_to_move;
    int50 pass_cnt;                     // number of consecutive passes played in a row most recently (by both players) 
    TTTCell* const board_1d;            // auxiliary pointer to facilitate 1d-array operations over the board

protected:
    int50 last_move;                    // 1d-position of the last move
    TTTPlayer winning_chain_color;      // indicate the player who has first made a winning chain; set to 'EMPTY' if the board does not contain a winning chain yet
    int50 winning_chain_map[BOARD_SIZE][BOARD_SIZE];
    
    int50 col_winning_chain;
    char dir_winning_chain;              // 'r' for row chain,'c' for column chain, 'd' for diagonal chain 

public:
    GameTTT() 
        : board_1d((TTTCell*)board) 
    {
        Reset();
    }
    virtual ~GameTTT() {}

    inline int50 I(int50 pos) {return pos/BOARD_SIZE;}
    inline int50 J(int50 pos) {return pos%BOARD_SIZE;}
    inline int50 ID(int50 i, int50 j) {return i*BOARD_SIZE+j;}


public:
    void Reset()
    {
        for(int50 i=0; i<BOARD_SIZE; i++) 
        for(int50 j=0; j<BOARD_SIZE; j++) 
        {
            board[i][j] = EMPTY;
            winning_chain_map[i][j] = 0;
        }

        player_to_move = BLACK;
        pass_cnt = 0;   
        last_move = -1;
        winning_chain_color = EMPTY;
        
    }

    void Play(IN_ int50 pos)
    {
        if(board_1d[pos] == EMPTY)
        {
            board_1d[pos] = player_to_move;  
            pass_cnt = 0;  
        }
        else
        {
            pass_cnt ++;  // play on an occupied cell is interpreted as a pass move
        } 

        player_to_move = (player_to_move == BLACK) ? WHITE : BLACK;
        last_move = pos;
    }

    // judge if game is over, in which case set 'winner' to be the winner, or to 'EMPTY' for draw game;
    // when the game is not over, how the routine will set 'winner' is undefined
    void Evaluate(OUT_ bool& game_over, OUT_ TTTPlayer& winner)
    {
        if(winning_chain_color != EMPTY)
        {
            game_over = true;
            winner = winning_chain_color;
        }
        else if(pass_cnt >= 2)
        {
            game_over = true;
            winner = EMPTY;
        }
        else if(last_move < 0 || last_move >= BOARD_SIZE*BOARD_SIZE)  
        { 
            // this should occur only at the starting position of the game
            for(int50 i=0; i<BOARD_SIZE; i++) for(int50 j=0; j<BOARD_SIZE; j++) assert(board[i][j] == EMPTY);

            game_over = false;  
            winner = EMPTY;
        }
        else // if neither a winning chain exists nor both players have passed and if we have a valid 'last_move'
        {
            // identify and mark all the new winning chains caused by the last move
            TTTPlayer color = board_1d[last_move];
            assert(color == BLACK || color == WHITE);
            int50 row_last = I(last_move);
            int50 col_last = J(last_move);

            int50 row_min = max(0, row_last - WINNING_CHAIN_SIZE +1);
            int50 row_max = min(row_last +WINNING_CHAIN_SIZE, BOARD_SIZE) -WINNING_CHAIN_SIZE;
            for(int50 r = row_min; r<= row_max; r++)
            {
                int50 cnt=0;
                for(int50 i=0; i<WINNING_CHAIN_SIZE; i++) cnt += (board[r+i][col_last] == color) ? 1 : 0;
                if(cnt == WINNING_CHAIN_SIZE)
                {
                    winning_chain_color = color;
                    for(int50 i=0; i<WINNING_CHAIN_SIZE; i++) winning_chain_map[r+i][col_last] += 1;    
                }
            }

            int50 col_min = max(0, col_last - WINNING_CHAIN_SIZE +1);
            int50 col_max = min(col_last +WINNING_CHAIN_SIZE, BOARD_SIZE) -WINNING_CHAIN_SIZE;
            for(int50 c = col_min; c<= col_max; c++)
            {
                int50 cnt=0;
                for(int50 j=0; j<WINNING_CHAIN_SIZE; j++) cnt += (board[row_last][c+j] == color) ? 1 : 0;
                if(cnt == WINNING_CHAIN_SIZE)
                {
                    winning_chain_color = color;
                    for(int50 j=0; j<WINNING_CHAIN_SIZE; j++) winning_chain_map[row_last][c+j] += 1;    
                }
            }

            row_min = (row_last <= col_last) ? row_min : 0;
            col_min = (col_last <= row_last) ? col_min : 0;
            for(int50 r=row_min, c=col_min; r<= row_last && c <= col_last; r++, c++)
            {
                if(r+WINNING_CHAIN_SIZE > BOARD_SIZE || c+WINNING_CHAIN_SIZE > BOARD_SIZE) break;
                
                int50 cnt=0;
                for(int50 s=0; s<WINNING_CHAIN_SIZE; s++) cnt += (board[r+s][c+s] == color) ? 1 : 0;
                if(cnt == WINNING_CHAIN_SIZE)
                {
                    winning_chain_color = color;
                    for(int50 s=0; s<WINNING_CHAIN_SIZE; s++) winning_chain_map[r+s][c+s] += 1;    
                }
            }

            for(row_min=row_last, col_max=col_last; row_min>0 && col_max<BOARD_SIZE-1; row_min--, col_max++) ;
            for(int50 r=row_min, c=col_max; r<= row_last && c >= col_last; r++, c--)
            {
                if(r+WINNING_CHAIN_SIZE-1 > BOARD_SIZE-1 || c-WINNING_CHAIN_SIZE+1 < 0) break;
                
                int50 cnt=0;
                for(int50 s=0; s<WINNING_CHAIN_SIZE; s++) cnt += (board[r+s][c-s] == color) ? 1 : 0;
                if(cnt == WINNING_CHAIN_SIZE)
                {
                    winning_chain_color = color;
                    for(int50 s=0; s<WINNING_CHAIN_SIZE; s++) winning_chain_map[r+s][c-s] += 1;    
                }
            }

            if(winning_chain_color != EMPTY)
            {
                game_over = true;
                winner = winning_chain_color;   
            }
            else
            {
                game_over = false;  
                winner = EMPTY;
            }
        }
    }

    void Print(FILE* fp=stdout)
    {
        fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, " %c  ", 'a'+j); fprintf(fp, "\n");
        fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, "----"); fprintf(fp, "\n");

        for(int50 i=0; i<BOARD_SIZE; i++)
        {
            printf("%6lld |", i);
            for(int50 j=0; j<BOARD_SIZE; j++)
            {
                if(winning_chain_map[i][j] > 0)
                {
                    assert(board[i][j] == winning_chain_color);
                    fprintf(fp, " %c |", (board[i][j] == BLACK)?'B':'W');
                }
                else if(board[i][j] != EMPTY)
                {
                    fprintf(fp, " %c |", (board[i][j] == BLACK)?'b':'w');
                }
                else
                {
                    fprintf(fp, "   |");
                }
            }
            fprintf(fp, "\n");
            fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, "----"); fprintf(fp, "\n");
        }

        fprintf(fp, "\n");
        fprintf(fp, "player_to_move = %s \t pass_cnt = %lld\n", (player_to_move == BLACK)?"BLACK":"WHITE", pass_cnt);
        fprintf(fp, "\n");
    }
};


enum TTTFeature
{
    NONE = 0,
    SELF = 1,
    OPPO = 2
};

template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
class Environment_TTT : public Environment<BOARD_SIZE*BOARD_SIZE*3, BOARD_SIZE*BOARD_SIZE>
{
public:
    static const int50 SIZE_PERCEPT_TTT  = BOARD_SIZE*BOARD_SIZE*3;
    static const int50 SIZE_ACTION_TTT  = BOARD_SIZE*BOARD_SIZE;

    GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE> game;
    bool game_over;
    TTTPlayer winner;  
    TTTPlayer agent_color;
    const TTTPlayer agent_color_setup;
    Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>* p_opp;

    bool verbose_mode;
    int50 round;


public:
    Environment_TTT(Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>* opponent, TTTPlayer agent_color =EMPTY, bool verbose =false) 
        : p_opp(opponent), agent_color_setup(agent_color), verbose_mode(verbose), round(0) 
    {
        Reset();
    }

	virtual ~Environment_TTT(){}

    // perform an opponent step inside the environment, using the internal agent of the environment 
    void PerformOpponentStep()
    {
        if(game_over)
        {
            // we will not make a real opponent move if the game is already over,
            // instead, we directly flip 'player_to_move' to skip this ply
            game.player_to_move = (game.player_to_move==BLACK)?WHITE:BLACK;
            return;
        }
            
        if(verbose_mode) game.Print();

        double x[BOARD_SIZE][BOARD_SIZE][3];
        double a[BOARD_SIZE][BOARD_SIZE];

        for(int50 i=0; i<BOARD_SIZE; i++) 
        for(int50 j=0; j<BOARD_SIZE; j++) 
        {
            if(game.board[i][j] == EMPTY)
            {
                x[i][j][NONE] = 1; x[i][j][SELF] = 0; x[i][j][OPPO] = 0;
            }
            else if(game.board[i][j] != agent_color) // the percept should be in the perspective of the the opponent player
            {
                x[i][j][NONE] = 0; x[i][j][SELF] = 1; x[i][j][OPPO] = 0;
            }
            else
            {
                x[i][j][NONE] = 0; x[i][j][SELF] = 0; x[i][j][OPPO] = 1;
            }
        }
        p_opp->TakeAction((double*)a, (double*)x, 0, false);

        int50 move = vec_argmax<double>((double*)a, BOARD_SIZE*BOARD_SIZE);
        game.Play(move);

        if(verbose_mode) system("cls");
    }

public:
    virtual void Reset()
    {
        game.Reset();
        game_over = false;
        winner = EMPTY;
        agent_color = (agent_color_setup != EMPTY) ? agent_color_setup : (rand()%2==0) ? BLACK : WHITE;
        round ++;

        if(verbose_mode) printf("Game %lld: agent plays %s\n\n", round, (agent_color==BLACK)?"BLACK":"WHITE");

        if(agent_color == WHITE)
        {
            PerformOpponentStep();
        }
    }

	virtual bool Update(OUT_ double* x, OUT_ double& r, OUT_ bool& fTerminal, IN_ double* a)
    {
        assert(game.player_to_move == agent_color);

        if(game_over)
        {
            Reset();
        }
        else
        {
            int50 move = vec_argmax<double>(a, BOARD_SIZE*BOARD_SIZE);
            game.Play(move);
            game.Evaluate(game_over, winner);
            PerformOpponentStep();
        }
        game.Evaluate(game_over, winner);

        assert(game.player_to_move == agent_color);
        fTerminal = game_over;
        r = (not game_over || winner == EMPTY) ? 0 : (winner == agent_color) ? 1 : -1; 
        for(int50 p=0; p<BOARD_SIZE*BOARD_SIZE; p++)
        {
            if(game.board_1d[p] == EMPTY)
            {
                x[p*3+NONE] = 1; x[p*3+SELF] = 0; x[p*3+OPPO] = 0;
            }
            else if(game.board_1d[p] == agent_color)
            {
                x[p*3+NONE] = 0; x[p*3+SELF] = 1; x[p*3+OPPO] = 0;
            }
            else
            {
                x[p*3+NONE] = 0; x[p*3+SELF] = 0; x[p*3+OPPO] = 1;
            }
        }

        return fTerminal;
    }
    
    virtual void Print(FILE* fp=stdout) 
    {
        game.Print(fp);
        if(game_over)
        {
            fprintf(fp, "Game is over. Winner is %s.\n\n", (winner == BLACK)?"BLACK":(winner==WHITE)?"WHITE":"NONE");
        }
    }

    virtual void Report(FILE* fp=stdout, int detail_level=1, bool reset=true) {}
};



template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
class AgentKB_TTT : public Agent<BOARD_SIZE*BOARD_SIZE*3, BOARD_SIZE*BOARD_SIZE>
{
public:
    AgentKB_TTT() {}
	virtual ~AgentKB_TTT(){}

    virtual bool Act(OUT_ double* a, IN_ double* x){return true;}

    virtual bool Learn(IN_ double* x, IN_ double* a, IN_ double r){return true;}

	virtual bool TakeAction(OUT_ double* a, IN_ double* x, IN_ double r, IN_ bool fTerminal=false)
	{
        FILE* fp = stdout;
		fprintf(fp, "\nr = %lf \t fTerminal = %s\n\n", r, (fTerminal)?"true":"false");
		
        // show board from the player's perspective
        double board[BOARD_SIZE][BOARD_SIZE][3];
        memcpy(board, x, sizeof(board));
        char symbol[3] = {' ', 'x', 'o'}; // 'x' for agent's piece, 'o' for opponent's piece
        
        fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, " %c  ", 'a'+j); fprintf(fp, "\n");
        fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, "----"); fprintf(fp, "\n");

        for(int50 i=0; i<BOARD_SIZE; i++)
        {
            printf("%6lld |", i);
            for(int50 j=0; j<BOARD_SIZE; j++)
            {
                int50 content = vec_argmax<double>(board[i][j], 3);
                fprintf(fp, " %c |", symbol[content]);
            }
            fprintf(fp, "\n\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, "----"); fprintf(fp, "\n");
        }
        fprintf(fp, "\n('%c' stands for SELF, '%c' stands for OPPONENT)\n\n", symbol[SELF], symbol[OPPO]);

        while(true)
        {
            printf("Input: ");
            int row;
            int col;
            char col_chr;
            if( scanf("%d%c", &row, &col_chr) != 2 ) continue;
            col = col_chr - 'a';
            if(row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) continue;

            int50 move = row*BOARD_SIZE + col;
            for(int i=0; i<BOARD_SIZE*BOARD_SIZE; i++) a[i] = 0;
            a[move] = 1;
            break;
        }

		return true;
	}

    virtual void Print(FILE* fp=stdout) {} 
    virtual void Report(FILE* fp=stdout, int50 detail_level=1, bool reset=true) {}
};


void ttt_test()
{
    static const int board_size = 15;
    static const int winning_chain_size = 5;
    AgentKB_TTT<board_size,winning_chain_size> agent;
    Environment_TTT<board_size,winning_chain_size> env(&agent, EMPTY, true);
    
    double x[board_size][board_size][3];
    double a[board_size][board_size];
    double r = 0;
    bool fTerminal = false;

    for(int50 i=0; i<board_size; i++) 
    for(int50 j=0; j<board_size; j++) 
    {
        if(env.game.board[i][j] == EMPTY)
        {
            x[i][j][NONE] = 1; x[i][j][SELF] = 0; x[i][j][OPPO] = 0;
        }
        else if(env.game.board[i][j] == env.agent_color) 
        {
            x[i][j][NONE] = 0; x[i][j][SELF] = 1; x[i][j][OPPO] = 0;
        }
        else
        {
            x[i][j][NONE] = 0; x[i][j][SELF] = 0; x[i][j][OPPO] = 1;
        }
    }

    while(true)
    {
        env.Print();
        agent.TakeAction((double*)a, (double*)x, r, fTerminal);
        system("cls");
        env.Update((double*)x, r, fTerminal, (double*)a);
    }

}

#endif
