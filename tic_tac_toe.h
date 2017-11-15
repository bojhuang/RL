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

public:
    GameTTT() 
        : board_1d((TTTCell*)board) 
    {
        Reset();
    }
    GameTTT(const GameTTT& game_rhs) 
        : board_1d((TTTCell*)board) 
    {
        Copy(game_rhs);
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

    void Copy(const GameTTT& game_rhs)
    {
        for(int50 i=0; i<BOARD_SIZE; i++) 
        for(int50 j=0; j<BOARD_SIZE; j++) 
        {
            board[i][j] = game_rhs.board[i][j];
            winning_chain_map[i][j] = game_rhs.winning_chain_map[i][j];
        }

        player_to_move = game_rhs.player_to_move;
        pass_cnt = game_rhs.pass_cnt;   
        last_move = game_rhs.last_move;
        winning_chain_color = game_rhs.winning_chain_color;
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

            int50 row_min = max2(0, row_last - WINNING_CHAIN_SIZE +1);
            int50 row_max = min2(row_last +WINNING_CHAIN_SIZE, BOARD_SIZE) -WINNING_CHAIN_SIZE;
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

            int50 col_min = max2(0, col_last - WINNING_CHAIN_SIZE +1);
            int50 col_max = min2(col_last +WINNING_CHAIN_SIZE, BOARD_SIZE) -WINNING_CHAIN_SIZE;
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

        if(not game_over)
        {
            assert(winner == EMPTY);

            bool board_is_full = true;
            for(int50 p=0; p<BOARD_SIZE*BOARD_SIZE; p++)
            {
                if(board_1d[p] == EMPTY)
                {
                    board_is_full = false;
                    break;
                }
            }
            game_over = board_is_full;
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




////
// Interface format:
// percept = (board[row][col][TTTCell], pass_cnt, player_to_move)
// action = (board[row][col])
////
#define SIZE_PERCEPT_TTT    (BOARD_SIZE*BOARD_SIZE*3+2)
#define SIZE_ACTION_TTT     (BOARD_SIZE*BOARD_SIZE)

template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
class Environment_TTT : public Environment<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>
{
public:
    GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE> game;
    bool game_over;
    TTTPlayer winner;  
    TTTPlayer agent_color;

    TTTPlayer agent_color_setup;
    Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>* p_opp;
    bool verbose_mode;
    int50 round;


public:
    Environment_TTT(Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>* opponent =NULL, TTTPlayer agent_color =EMPTY, bool verbose =false) 
        : p_opp(opponent), agent_color_setup(agent_color), verbose_mode(verbose), round(0) 
    {
        Reset();
    }

	virtual ~Environment_TTT(){}


	virtual bool Update(OUT_ double* x, OUT_ double& r, OUT_ bool& fTerminal, IN_ double* a)
    {
        assert(game_over || game.player_to_move == agent_color);

        if(game_over)
        {
            Reset();
        }
        else
        {
            int50 move = vec_argmax<double>(a, BOARD_SIZE*BOARD_SIZE);
            game.Play(move);
            
            game.Evaluate(game_over, winner);
            if(not game_over)
            {
                PerformOpponentStep();
            }
        }
        game.Evaluate(game_over, winner);

        assert(game_over || game.player_to_move == agent_color);

        if(game_over)
        {
            // inform the opponent agent immediately if game is over 
            // (note that this could reverse the playing order temporarily during the "game over" phase)
            PerformOpponentStep();
        }

        fTerminal = game_over;
        r = (not game_over || winner == EMPTY) ? 0 : (winner == agent_color) ? 1 : -1; 
        GenPercept(x);
        return fTerminal;
    }
    
public:
    virtual void Reset()
    {
        game.Reset();
        game_over = false;
        winner = EMPTY;
        agent_color = (agent_color_setup != EMPTY) ? agent_color_setup : (rand()%2==0) ? BLACK : WHITE;
        round ++;

        if(verbose_mode) printf("Game %lld: the agent plays %s\n\n", round, (agent_color==BLACK)?"BLACK":"WHITE");

        if(agent_color == WHITE)
        {
            PerformOpponentStep();
        }
    }

    // perform an opponent step inside the environment, using the 'p_opp' agent of the environment 
    void PerformOpponentStep()
    {    
        if(verbose_mode) 
        {
            system("cls");
            printf("[ Opponent (%s) 's Turn ]\n\n", (agent_color!=BLACK)?"BLACK":"WHITE");
            game.Print();
        }

        double x[SIZE_PERCEPT_TTT];
        double a[SIZE_ACTION_TTT];

        bool fTerminal = game_over;
        double r = (not game_over || winner == EMPTY) ? 0 : (winner != agent_color) ? 1 : -1;       // it's a zero-sum game 
        GenPercept(x);
        p_opp->TakeAction(a, x, r, fTerminal);

        // update the game if the game is going on;
        // oterhwise if the game has been over, we keep the game state frozen before resetting it (later at somewhere else)
        if(not game_over)
        {
            int50 move = vec_argmax<double>(a, SIZE_ACTION_TTT);
            game.Play(move);
        }
    }

     void GenPercept(double* x) const
    {
        int50 p;
        for(p=0; p<BOARD_SIZE*BOARD_SIZE*3; p+=3) 
        {
            x[p + EMPTY] = (game.board_1d[p/3] == EMPTY) ? 1 : 0;
            x[p + BLACK] = (game.board_1d[p/3] == BLACK) ? 1 : 0;
            x[p + WHITE] = (game.board_1d[p/3] == WHITE) ? 1 : 0;
        }
        x[p++] = game.pass_cnt;
        x[p++] = game.player_to_move;
        assert(p == SIZE_PERCEPT_TTT);
    }

public:
    virtual void Print(FILE* fp=stdout) 
    {
        game.Print(fp);
        if(game_over)
        {
            fprintf(fp, "Game is over. Winner is %s.\n\n", (winner == BLACK)?"BLACK":(winner==WHITE)?"WHITE":"NONE");
        }
    }

    virtual void Report(FILE* fp=stdout, int detail_level=1, bool reset=true) 
    {
        p_opp->Report(fp, detail_level, reset);
    }
};





enum TTTFeature
{
    NONE = 0,
    SELF = 1,
    OPPO = 2
};

//
#define SIZE_PERCEPT_T3W       (BOARD_SIZE*BOARD_SIZE*3)
#define SIZE_ACTION_T3W        SIZE_ACTION_TTT

template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
class Agent_TTTWrapper : public Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>
{
public:
    Agent<SIZE_PERCEPT_T3W, SIZE_ACTION_T3W>* pAgent;
    bool first_turn_in_game;
    double r_last_game;
    bool verbose_mode;

public:
    Agent_TTTWrapper(Agent<SIZE_PERCEPT_T3W, SIZE_ACTION_T3W>* pAgent_rhs, bool verbose =false)
        : pAgent(pAgent_rhs), first_turn_in_game(true), r_last_game(0), verbose_mode(verbose)
    {}

	virtual ~Agent_TTTWrapper(){}


	virtual bool TakeAction(OUT_ double* a, IN_ double* x, IN_ double r, IN_ bool fTerminal=false)
	{
        int50 pass_cnt = x[BOARD_SIZE*BOARD_SIZE*3];
        TTTPlayer player_to_move = (x[BOARD_SIZE*BOARD_SIZE*3+1] == BLACK) ? BLACK : (x[BOARD_SIZE*BOARD_SIZE*3+1] == WHITE) ? WHITE : EMPTY;
        assert(player_to_move == BLACK || player_to_move == WHITE);
        assert(pass_cnt < 2 || fTerminal == true);
        assert(not (first_turn_in_game && fTerminal));

        // convert board to first-person perspective
        double x_board[SIZE_PERCEPT_T3W];
        for(int50 p=0; p<BOARD_SIZE*BOARD_SIZE; p++)
        {
            TTTCell color = (TTTCell)vec_argmax<double>(x+3*p, 3);
            x_board[3*p+NONE] = (color == EMPTY) ? 1 : 0;
            x_board[3*p+SELF] = (color == player_to_move) ? 1 : 0;
            x_board[3*p+OPPO] = (color != player_to_move) ? 1 : 0;   
        }

        if(first_turn_in_game)
        {
            pAgent->TakeAction(a, x_board, r_last_game, true);
        }
        else if(not fTerminal)
        {
            pAgent->TakeAction(a, x_board, r, false);
        }
        else // if the game has ended, with at least one action performed by the agent in the game
        {
            // skip the "confirmation" action (so that it won't confuse the real agent) 
        }

        if(verbose_mode)
        {
            if(first_turn_in_game || not fTerminal)
            {
                // show the received x_board ('x' stands for agent's piece, 'o' stands for opponent's piece)
                char symbol[3] = {' ', 'x', 'o'};
                FILE* fp = stdout;

                //system("cls");
                fprintf(fp, "\n---------- Agent_TTTWrapper ------------\n\n");
                fprintf(fp, "r = %.0lf \t fTerminal = %s \t (first_turn = %s \t r_last_game = %.0lf)\n\n", 
                    (first_turn_in_game) ? r_last_game : (not fTerminal) ? r : -1000, 
                    (first_turn_in_game) ? "true" : (not fTerminal) ? "false" : "confirmation turn",
                    (first_turn_in_game) ? "true" : "false",
                    r_last_game );

                fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, " %c  ", 'a'+j); fprintf(fp, "\n");
                fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, "----"); fprintf(fp, "\n");

                for(int50 i=0; i<BOARD_SIZE; i++)
                {
                    printf("%6lld |", i);
                    for(int50 j=0; j<BOARD_SIZE; j++)
                    {
                        int50 feature = vec_argmax(x_board + 3*(i*BOARD_SIZE+j), 3);
                        fprintf(fp, " %c |", symbol[feature]);
                    }
                    fprintf(fp, "\n\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, "----"); fprintf(fp, "\n");
                }
                //fprintf(fp, "\n('%c' stands for SELF, '%c' stands for OPPONENT)\n\n", symbol[SELF], symbol[OPPO]);

                fprintf(fp, "\naction = %lld\n\n", vec_argmax<double>(a, SIZE_ACTION_T3W));

                system("pause");
                //while(getchar() == '\n') ;
            }
        }

        if(fTerminal)
        {
            // since the real agent does not take action at the terminal position of a game,
            // we need to delay both the "end-of-episode" signal and the "end-of-episode" reward to the first turn of the agent in the next game
            first_turn_in_game = true;
            r_last_game = r;
        }
        else
        {
            // a non-terminal position cannot directly followed by a "first-turn" position (of the next game)
            first_turn_in_game = false;
            r_last_game = 0; 
        }

		return true;
	}

    virtual bool Act(OUT_ double* a, IN_ double* x)
    {
        return false;
    }
	virtual bool Learn(IN_ double* x, IN_ double* a, IN_ double r)
    {
        return false;
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




template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
class AgentKB_TTT : public Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>
{
public:
    AgentKB_TTT() {}
	virtual ~AgentKB_TTT(){}

	virtual bool TakeAction(OUT_ double* a, IN_ double* x, IN_ double r, IN_ bool fTerminal=false)
	{
        FILE* fp = stdout;
		fprintf(fp, "\nr = %lf \t fTerminal = %s\n\n", r, (fTerminal)?"true":"false");
		 
        // show the board from this agent's perspective;
        // 'x' for agent's piece, 'o' for opponent's piece
        char symbol[3] = {' ', 'x', 'o'};
        TTTFeature board[BOARD_SIZE][BOARD_SIZE];
        int50 pass_cnt;
        TTTPlayer player_to_move;

        pass_cnt = x[BOARD_SIZE*BOARD_SIZE*3];
        player_to_move = (x[BOARD_SIZE*BOARD_SIZE*3+1] == BLACK) ? BLACK : (x[BOARD_SIZE*BOARD_SIZE*3+1] == WHITE) ? WHITE : EMPTY;
        assert(player_to_move == BLACK || player_to_move == WHITE);
        for(int50 p=0; p<BOARD_SIZE*BOARD_SIZE; p++)
        {
            TTTCell color = (TTTCell)vec_argmax<double>(x+3*p, 3);
            board[p/BOARD_SIZE][p%BOARD_SIZE] = (color==EMPTY) ? NONE : (color == player_to_move) ? SELF : OPPO;       
        } 
  
        fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, " %c  ", 'a'+j); fprintf(fp, "\n");
        fprintf(fp, "\t"); for(int50 j=0; j<BOARD_SIZE; j++) fprintf(fp, "----"); fprintf(fp, "\n");

        for(int50 i=0; i<BOARD_SIZE; i++)
        {
            printf("%6lld |", i);
            for(int50 j=0; j<BOARD_SIZE; j++)
            {
                fprintf(fp, " %c |", symbol[board[i][j]]);
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
            char buf[100];
            scanf("%s", buf);
            if(buf[0] >= '0' && buf[0] <= '9')
            {
                if( sscanf(buf, "%d%c", &row, &col_chr) != 2 ) continue;   
            }
            else
            {
                if( sscanf(buf, "%c%d", &col_chr, &row) != 2 ) continue;
            }
            
            col = col_chr - 'a';
            if(row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) continue;

            int50 move = row*BOARD_SIZE + col;
            if(board[row][col] != NONE)
            {
                printf("play [%s] will pass the turn, are you sure? (y/n) ", buf);
                char confirm_chr = 0;
                while(confirm_chr != 'y' && confirm_chr != 'n') confirm_chr = getchar();
                if(confirm_chr == 'n') continue;
                assert(confirm_chr == 'y');
            }

            for(int i=0; i<BOARD_SIZE*BOARD_SIZE; i++) a[i] = 0;
            a[move] = 1;
            break;
        }

		return true;
	}

    virtual bool Act(OUT_ double* a, IN_ double* x){return true;}
    virtual bool Learn(IN_ double* x, IN_ double* a, IN_ double r){return true;}
    virtual void Print(FILE* fp=stdout) {} 
    virtual void Report(FILE* fp=stdout, int50 detail_level=1, bool reset=true) {}
};



template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
class Agent_TTTMinimax : public Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>
{
public:
    class PositionRecord
    {
    public:
        bool settled;
        TTTPlayer winner;

        PositionRecord() : settled(false), winner(EMPTY) {}
    };
    
    class TransportationTable
    {
    protected:
        int50 n_settled;
        PositionRecord* s; 
        
    public:
        TransportationTable()
        {
            n_settled = 0;
            if(BOARD_SIZE > 4) 
            {
                printf("[Agent_TTTMinimax Error] board size (%lld) is too large to create the transportation table.\n", BOARD_SIZE);
                s = NULL;
            }
            else
            {
                s = new PositionRecord[((int50)1)<<(BOARD_SIZE*BOARD_SIZE*2+2)];
            }    
        }

        virtual ~TransportationTable() 
        {
            if(s != NULL) delete[] s;
        }

        PositionRecord* GetRecord(const GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE>& game)
        {
            int50 id = GetID(game);
            return &s[id];
        }

        void SetRecord(const GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE>& game, TTTPlayer winner, PositionRecord* pRecord_from_user =NULL)
        {
            PositionRecord* pRecord = (pRecord_from_user) ? pRecord_from_user : GetRecord(game);
            
            if(pRecord->settled == false) 
                n_settled ++;
            
            pRecord->winner = winner;
            pRecord->settled = true;
        }
        
        // even id's for BLACK's turns, odd id's for WHITE's turns
        int50 GetID(const GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE>& game)
        {
            // terminal position should not be buffered
            assert(game.pass_cnt == 0 || game.pass_cnt == 1);
            assert(game.player_to_move == BLACK || game.player_to_move == WHITE);

            int50 id = 0;
            for(int50 p=0; p<BOARD_SIZE*BOARD_SIZE; p++)
            {
                id = id << 2;
                id += game.board_1d[p];
            }

            id = id << 1;
            id += game.pass_cnt;
            
            id = id << 1;
            id += (game.player_to_move == BLACK) ? 0 : 1;

            return id;
        }
    };
    
public:
    int50 search_cnt;
    TransportationTable tt;
    
public:
    Agent_TTTMinimax() {}
    virtual ~Agent_TTTMinimax() {}

    virtual bool TakeAction(OUT_ double* a, IN_ double* x, IN_ double r, IN_ bool fTerminal=false)
	{
        if(fTerminal == true) 
        {
            if(r < 0) 
            {
                printf("[Error] Agent_TTTMinimax lost a game (r=%lf).\n", r);
                getchar();
            }
            return true;
        }

        int50 pass_cnt = x[BOARD_SIZE*BOARD_SIZE*3];
        TTTPlayer player_to_move = (x[BOARD_SIZE*BOARD_SIZE*3+1] == BLACK) ? BLACK : (x[BOARD_SIZE*BOARD_SIZE*3+1] == WHITE) ? WHITE : EMPTY;
        assert(player_to_move == BLACK || player_to_move == WHITE);
        assert(pass_cnt < 2 || fTerminal == true);

        GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE> s;
        s.pass_cnt = pass_cnt;
        s.player_to_move = player_to_move;
        for(int50 p=0; p<BOARD_SIZE*BOARD_SIZE; p++) s.board_1d[p] = (TTTCell)vec_argmax<double>(x+3*p, 3); 
        
        std::vector<int50> winning_moves;
        std::vector<int50> draw_moves;
        std::vector<int50> losing_moves;
        search_cnt = 0;

        for(int50 action = 0; action<BOARD_SIZE*BOARD_SIZE; action++)
        {
            GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE> s_next(s);
            s_next.Play(action);

            TTTPlayer winner = Minimax(s_next);
            
            if(winner == EMPTY) 
            {
                draw_moves.push_back(action);
            }
            else if(winner == s.player_to_move)
            {
                winning_moves.push_back(action);
            }
            else
            {
                losing_moves.push_back(action);
            }
        }

        int50 opt_move;
        if(winning_moves.size() > 0) 
        {
            opt_move = winning_moves[ rand()%winning_moves.size() ];
        }
        else if(draw_moves.size() > 0)
        {
            opt_move = draw_moves[ rand()%draw_moves.size() ];
        }
        else
        {
            assert(losing_moves.size() > 0);
            opt_move = losing_moves[ rand()%losing_moves.size() ];
        }

        for(int i=0; i<SIZE_ACTION_TTT; i++) a[i] = 0;
        a[opt_move] = 1;

        return true;
    }

    TTTPlayer Minimax(GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE>& s)
    {
        search_cnt ++;
        //if(search_cnt%1000 == 0) printf("#nodes searched = %lld\n", search_cnt);

        TTTPlayer winner;
        bool game_over;
        s.Evaluate(game_over, winner);
        if(game_over)
        {
            return winner;
        }

        int50 id = tt.GetID(s);
        PositionRecord* pRecord = tt.GetRecord(s); 
        if(pRecord != NULL && pRecord->settled == true) return pRecord->winner;

        TTTPlayer self = s.player_to_move;
        TTTPlayer oppo = (s.player_to_move == BLACK) ? WHITE : BLACK;
        
        winner = oppo;
        bool pass_move_tested = false;

        for(int50 action = 0; action<BOARD_SIZE*BOARD_SIZE; action++)
        {
            if(s.board_1d[action] != EMPTY && pass_move_tested) continue;
            if(s.board_1d[action] != EMPTY) pass_move_tested = true;

            GameTTT<BOARD_SIZE, WINNING_CHAIN_SIZE> s_next(s);
            s_next.Play(action);
            TTTPlayer w_next = Minimax(s_next);
            winner = (winner == oppo) ? w_next : (winner == EMPTY && w_next == self) ? w_next : winner;
        }

        tt.SetRecord(s, winner, pRecord);

        return winner;
    }

    virtual bool Act(OUT_ double* a, IN_ double* x)
    {
        return false;
    }
	virtual bool Learn(IN_ double* x, IN_ double* a, IN_ double r)
    {
        return false;
    }
    virtual void Print(FILE* fp=stdout) 
    {
        
    }
    virtual void Report(FILE* fp=stdout, int50 detail_level=1, bool reset=true) 
    {
       
    }
};


template<int50 BOARD_SIZE, int50 WINNING_CHAIN_SIZE>
void ttt_test(Agent<SIZE_PERCEPT_TTT, SIZE_ACTION_TTT>* pAgent_test =NULL, TTTPlayer test_agent_color = EMPTY)
{
    AgentKB_TTT<BOARD_SIZE,WINNING_CHAIN_SIZE> agent_kb;
    pAgent_test = (pAgent_test) ? (pAgent_test) : (&agent_kb);
    TTTPlayer agent_color = (test_agent_color == BLACK) ? WHITE : (test_agent_color == WHITE) ? BLACK : EMPTY;
    Environment_TTT<BOARD_SIZE,WINNING_CHAIN_SIZE> env(pAgent_test , agent_color, true );

    double x[SIZE_PERCEPT_TTT];
    double a[SIZE_ACTION_TTT];
    double r = 0;
    bool fTerminal = false;
    env.GenPercept(x);
    
    while(true)
    {
        system("cls");
        printf("[ Agent (%s)'s Turn ]\n\n", (env.agent_color==BLACK)?"BLACK":"WHITE");
        env.Print();
        agent_kb.TakeAction((double*)a, (double*)x, r, fTerminal);
        env.Update((double*)x, r, fTerminal, (double*)a);
    }

}

#endif
