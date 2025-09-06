#include "/home/codeleaded/System/Static/Library/WindowEngine1.0.h"
#include "/home/codeleaded/System/Static/Library/NeuralNetwork.h"
#include "/home/codeleaded/System/Static/Library/Thread.h"

#define PADDLE_1_POSX       0.05f
#define PADDLE_2_POSX       0.95f
#define PADDLE_WIDTH        0.025f
#define PADDLE_HEIGHT       0.18f
#define PADDLE_SPEED        0.25f

#define BALL_WIDTH          0.0125f
#define BALL_HEIGHT         0.025f
#define BALL_STARTSPEED     -0.1f
#define SPEED_ACCSPEED      0.05f

#define PADDLE_1_COLOR      BLUE
#define PADDLE_2_COLOR      RED
#define BALL_COLOR          WHITE

#define NN_LEARNRATE        0.0001f
#define NN_DELTA            0.99f
#define NN_DELAY            1
#define NN_PATH             "./data/Model"
#define NN_PATHTYPE         "nnalx"
#define NN_INPUTS           4
#define NN_OUTPUTS          3
#define TIME_STEP           0.05f

#define NN_GENS             10




typedef struct PongObject{
    Vec2 p;
    Vec2 d;
    Vec2 v;
    Pixel c;
} PongObject;

PongObject PongObject_New(Vec2 p,Vec2 d,Vec2 v,Pixel c){
    PongObject po;
    po.p = p;
    po.d = d;
    po.v = v;
    po.c = c;
    return po;
}
int PongObject_Update(PongObject* po,float t){
    po->p = Vec2_Add(po->p,Vec2_Mulf(po->v,t));

    if(po->p.x < -po->d.x){
        return 2;
    }
    if(po->p.x > 1.0f){
        return 1;
    }
    if(po->p.y < 0.0f){
        po->p.y = 0.0f;
        po->v.y *= -1.0f;
    }
    if(po->p.y > 1.0f - po->d.y){
        po->p.y = 1.0f - po->d.y;
        po->v.y *= -1.0f;
    }
    return 0;
}
void PongObject_Collision(PongObject* po,PongObject* target){
    if(Overlap_Rect_Rect((Rect){ po->p,po->d },(Rect){ target->p,target->d })){
        const float s = Vec2_Mag(target->v);
        const float dx = F32_Sign(target->p.x - po->p.x);
        
        const float pm = po->p.x + po->d.x * 0.5f;
        const float offset = dx * 0.5f * (po->d.x + target->d.x);

        target->p.x = pm + offset - 0.5f * target->d.x;
        target->v.x *= -1.0f;
        target->v = Vec2_Mulf(Vec2_Norm(target->v),s + SPEED_ACCSPEED);
        target->v.y += F32_Sign(target->v.y) * F32_Abs(po->v.y) * 0.1f;
        target->v.y = F32_Abs(target->v.y) * F32_Sign(po->v.y);
    }
}
void PongObject_Render(PongObject* po){
    RenderRect(po->p.x * GetWidth(),po->p.y * GetHeight(),po->d.x * GetWidth(),po->d.y * GetHeight(),po->c);
}

NeuralReward PongObject_Step_Update(PongObject* po,float t){
    po->p = Vec2_Add(po->p,Vec2_Mulf(po->v,t));

    if(po->p.x < -po->d.x){
        return 10.0f;
    }
    if(po->p.x > 1.0f){
        return -1.0f;
    }
    if(po->p.y < 0.0f){
        po->p.y = 0.0f;
        po->v.y *= -1.0f;
    }
    if(po->p.y > 1.0f - po->d.y){
        po->p.y = 1.0f - po->d.y;
        po->v.y *= -1.0f;
    }
    return 0.0f;
}
NeuralReward PongObject_Step_Collision(PongObject* po,PongObject* target){
    if(Overlap_Rect_Rect((Rect){ po->p,po->d },(Rect){ target->p,target->d })){
        const float s = Vec2_Mag(target->v);
        const float dx = F32_Sign(target->p.x - po->p.x);
        
        const float pm = po->p.x + po->d.x * 0.5f;
        const float offset = dx * 0.5f * (po->d.x + target->d.x);

        target->p.x = pm + offset - 0.5f * target->d.x;
        target->v.x *= -1.0f;
        target->v = Vec2_Mulf(Vec2_Norm(target->v),s + SPEED_ACCSPEED);
        target->v.y += F32_Sign(target->v.y) * F32_Abs(po->v.y) * 0.1f;
        target->v.y = F32_Abs(target->v.y) * F32_Sign(po->v.y);
        return 1.0f;
    }
    return 0;
}


typedef struct PongInfos{
    NeuralType balldx;
    NeuralType bally;
    NeuralType ballvy;
    NeuralType paddle1y;
    NeuralType paddle2y;
    // other step values
    NeuralType ballx;
    NeuralType ballvx;
} PongInfos;

PongInfos PongInfos_New(
    NeuralType ballx,
    NeuralType bally,
    NeuralType ballvx,
    NeuralType ballvy,
    NeuralType paddle1x,
    NeuralType paddle1y,
    NeuralType paddle2x,
    NeuralType paddle2y
){
    PongInfos pi;
    pi.balldx = F32_Abs(ballx - paddle2x);
    pi.bally = bally;
    pi.ballvy = ballvy;
    pi.paddle1y = paddle1y;
    pi.paddle2y = paddle2y;
    // ----------------------
    pi.ballx = ballx;
    pi.ballvx = ballvx;
    return pi;
}

typedef struct PongGame{
    PongObject paddle1;
    PongObject paddle2;
    PongObject ball;
    int score1;
    int score2;
    NeuralReward aireward;
} PongGame;

void PongGame_Reset(PongGame* pg){
    pg->paddle1.p.y = 0.5f - PADDLE_HEIGHT * 0.5f;
    pg->paddle2.p.y = 0.5f - PADDLE_HEIGHT * 0.5f;
    
    pg->ball.p.x = 0.5f - BALL_WIDTH * 0.5f;
    pg->ball.p.y = 0.5f - BALL_HEIGHT * 0.5f;
    //ball.v.x = BALL_STARTSPEED;
    //ball.v.y = 0.0f;

    float a = Random_f64_MinMax(0.0f,2.0f * F32_PI);
    a = F32_Border(a,0.25f * F32_PI,0.75f * F32_PI);
    a = F32_Border(a,1.25f * F32_PI,1.75f * F32_PI);

    pg->ball.v = Vec2_Mulf(Vec2_OfAngle(a),BALL_STARTSPEED);
}
PongGame PongGame_New(){
    PongGame pg;
    pg.paddle1 = PongObject_New((Vec2){ PADDLE_1_POSX,0.0f },(Vec2){ PADDLE_WIDTH,PADDLE_HEIGHT },(Vec2){ 0.0f,0.0f },PADDLE_1_COLOR);
    pg.paddle2 = PongObject_New((Vec2){ PADDLE_2_POSX,0.0f },(Vec2){ PADDLE_WIDTH,PADDLE_HEIGHT },(Vec2){ 0.0f,0.0f },PADDLE_2_COLOR);
    pg.ball = PongObject_New((Vec2){ 0.0f,0.0f },(Vec2){ BALL_WIDTH,BALL_HEIGHT },(Vec2){ 0.0f,0.0f },BALL_COLOR);
    pg.score1 = 0;
    pg.score2 = 0;
    pg.aireward = 0.0f;
    PongGame_Reset(&pg);
    return pg;
}
void PongGame_Update(PongGame* pg,float ElapsedTime){
    PongObject_Update(&pg->paddle1,ElapsedTime);
    PongObject_Update(&pg->paddle2,ElapsedTime);
    PongObject_Update(&pg->ball,ElapsedTime);

    PongObject_Collision(&pg->paddle1,&pg->ball);
    PongObject_Collision(&pg->paddle2,&pg->ball);
}
void PongGame_State(PongGame* pg,RLNeuralNetwork* nn,DecisionState* ds){
    PongInfos pi_before = PongInfos_New(
        pg->ball.p.x + pg->ball.d.x * 0.5f,
        pg->ball.p.y + pg->ball.d.y * 0.5f,
        pg->ball.v.x,
        pg->ball.v.y,
        pg->paddle1.p.x + pg->paddle1.d.x * 0.5f,
        pg->paddle1.p.y + pg->paddle1.d.y * 0.5f,
        pg->paddle2.p.x + pg->paddle2.d.x * 0.5f,
        pg->paddle2.p.y + pg->paddle2.d.y * 0.5f
    );
    DecisionState_SetBefore(ds,(NeuralType*)&pi_before);
}
void PongGame_Step(PongGame* pg,RLNeuralNetwork* nn,DecisionState* ds,int d){
    PongInfos pi_before = PongInfos_New(
        pg->ball.p.x + pg->ball.d.x * 0.5f,
        pg->ball.p.y + pg->ball.d.y * 0.5f,
        pg->ball.v.x,
        pg->ball.v.y,
        pg->paddle1.p.x + pg->paddle1.d.x * 0.5f,
        pg->paddle1.p.y + pg->paddle1.d.y * 0.5f,
        pg->paddle2.p.x + pg->paddle2.d.x * 0.5f,
        pg->paddle2.p.y + pg->paddle2.d.y * 0.5f
    );
    DecisionState_SetBefore(ds,(NeuralType*)&pi_before);

    NeuralReward aireward = 0.0f;

    if(d==0)      pg->paddle2.v.y = -PADDLE_SPEED;
    else if(d==1) pg->paddle2.v.y = PADDLE_SPEED;
    else if(d==2) pg->paddle2.v.y = 0.0f;
    
    aireward += PongObject_Step_Update(&pg->paddle1,TIME_STEP);
    aireward += PongObject_Step_Update(&pg->paddle2,TIME_STEP);
    aireward += PongObject_Step_Update(&pg->ball,TIME_STEP);
    aireward += PongObject_Step_Collision(&pg->paddle1,&pg->ball);
    aireward += PongObject_Step_Collision(&pg->paddle2,&pg->ball);

    // aireward += (1.0f - 
    //     //Vec2_Mag(Vec2_Sub(
    //     //    Vec2_Add(paddle2.p, Vec2_Mulf(paddle2.d,0.5f)),
    //     //    Vec2_Add(ball.p,    Vec2_Mulf(ball.d,   0.5f))
    //     //
    //     (paddle2.p.y  + paddle2.d.y   * 0.5f) -
    //     (ball.p.y     + ball.d.y      * 0.5f)
    // ) * 0.5f * window.ElapsedTime;

    const Vec2 pm = Vec2_Add(pg->paddle2.p, Vec2_Mulf(pg->paddle2.d,0.5f)); 
    const Vec2 bm = Vec2_Add(pg->ball.p,    Vec2_Mulf(pg->ball.d,   0.5f)); 
    const Vec2 dp = Vec2_Sub(pm,bm);
    if(F32_Sign(dp.y) == F32_Sign(pg->ball.v.y))
        aireward += (1.0f - dp.y) * 0.5f * window.ElapsedTime;

    PongInfos pi_after = PongInfos_New(
        pg->ball.p.x + pg->ball.d.x * 0.5f,
        pg->ball.p.y + pg->ball.d.y * 0.5f,
        pg->ball.v.x,
        pg->ball.v.y,
        pg->paddle1.p.x + pg->paddle1.d.x * 0.5f,
        pg->paddle1.p.y + pg->paddle1.d.y * 0.5f,
        pg->paddle2.p.x + pg->paddle2.d.x * 0.5f,
        pg->paddle2.p.y + pg->paddle2.d.y * 0.5f
    );
    DecisionState_SetAfter(ds,(NeuralType*)&pi_after);
    DecisionState_SetReward(ds,aireward);
}
void PongGame_Undo(PongGame* pg,RLNeuralNetwork* nn,DecisionState* ds){
    PongInfos* pi = (PongInfos*)ds->before;
    pg->ball.p.y = pi->bally;
    pg->ball.v.y = pi->ballvy;
    pg->paddle1.p.y = pi->paddle1y;
    pg->paddle2.p.y = pi->paddle2y;
    // ----------------------
    pg->ball.p.x = pi->ballx;
    pg->ball.v.x = pi->ballvx;
}
void PongGame_Render(PongGame* pg){
    PongObject_Render(&pg->paddle1);
    PongObject_Render(&pg->paddle2);
    PongObject_Render(&pg->ball);

    String str = String_Format("%d : %d",pg->paddle1,pg->score2);
    RenderCStrSize(str.Memory,str.size,(GetWidth() - str.size * GetAlxFont()->CharSizeX) * 0.5f,0.0f,WHITE);
    String_Free(&str);
}


char training = 0;
char ai = 0;
Neat nt;
AlxFont font;

void NeuralNetwork_Render(NeuralNetwork* nn){
    for(int i = 0;i<nn->layers.size;i++){
        NeuralLayer* nl = (NeuralLayer*)Vector_Get(&nn->layers,i);
        
        for(int j = 0;j<nl->count;j++){
            const int dx = 400.0f;
            const int x = i * dx;
            const int y = j * font.CharSizeY * 3;

            RenderRect(x,y,100.0f,font.CharSizeY * 2,GREEN);
            
            String str = String_Format("%f",nl->values[j]);
            RenderCStrSizeAlxFont(&font,str.Memory,str.size,x,y,GRAY);
            String_Free(&str);
        
            if(nl->precount > 0){
                str = String_Format("%f",nl->biases[j]);
                RenderCStrSizeAlxFont(&font,str.Memory,str.size,x,y + font.CharSizeY,GRAY);
                String_Free(&str);
            }
        
            const int max = 3;
            const int count = nl->precount < max ? nl->precount : max;
            for(int k = 0;k<count;k++){
                if(nl->weights && nl->weights[j]){
                    str = String_Format("%f",nl->weights[j][k]);
                    RenderCStrSizeAlxFont(&font,str.Memory,str.size,x - dx * 0.5f,y + k * font.CharSizeY,GRAY);
                    String_Free(&str);
                }
            }
        }
    }
}

void Setup(AlxWindow* w){
    RGA_Set(Time_Nano());
    
    font = AlxFont_MAKE_YANIS(12,12);

    PongGame envs[NN_GENS];
    for(int i = 0;i<NN_GENS;i++)
        envs[i] = PongGame_New();

    nt = Neat_New(
        NN_GENS,
        NN_INPUTS,
        NN_OUTPUTS,
        sizeof(PongInfos) / sizeof(NeuralType),
        (void*)PongGame_State,
        (void*)PongGame_Step,
        (void*)PongGame_Undo,
        10.0f,
        sizeof(PongGame),
        envs,
        NULL
    );
}
void Update(AlxWindow* w){
    ReloadAlxFont(GetWidth() * 0.025f,GetHeight() * 0.05f);

    if(Stroke(ALX_KEY_T).PRESSED){
        training = !training;
    }
    if(Stroke(ALX_KEY_Z).PRESSED){
        ai = ai + 1;
        if(ai >= NN_GENS) ai = 0;
    }

    Neat_Update(&nt,NN_DELTA,NN_LEARNRATE);

    for(int i = 0;i<nt.gensize;i++){
        RLNeuralNetworkEnv* rlnnenv = nt.rlnnenvs + i;
        PongGame* pg = (PongGame*)rlnnenv->env;

        const float mb = pg->ball.p.y + pg->ball.d.y * 0.5f;
        const float m1 = pg->paddle1.p.y + pg->paddle1.d.y * 0.5f + pg->paddle1.d.y * 0.2f * Random_f64_MinMax(-1.0f,1.0f);
        const float dy = mb - m1;
        pg->paddle1.v.y = PADDLE_SPEED * F32_Sign(dy);

        PongInfos pi = PongInfos_New(
            pg->ball.p.x + pg->ball.d.x * 0.5f,
            pg->ball.p.y + pg->ball.d.y * 0.5f,
            pg->ball.v.x,
            pg->ball.v.y,
            pg->paddle1.p.x + pg->paddle1.d.x * 0.5f,
            pg->paddle1.p.y + pg->paddle1.d.y * 0.5f,
            pg->paddle2.p.x + pg->paddle2.d.x * 0.5f,
            pg->paddle2.p.y + pg->paddle2.d.y * 0.5f
        );
        NeuralNetwork_Calc(&rlnnenv->rlnn.nn,(NeuralType*)&pi);

        int d = RLNeuralNetwork_DoNextDecision(&rlnnenv->rlnn);
        if(d==0)      pg->paddle2.v.y = -PADDLE_SPEED;
        else if(d==1) pg->paddle2.v.y = PADDLE_SPEED;
        else if(d==2) pg->paddle2.v.y = 0.0f;
        else          printf("Error in %d!\n",i);

        //const int state = (int)Random_u32_MinMax(0,3) - 1;
        //pg->paddle2.v.y = PADDLE_SPEED * state;

        PongGame_Update(pg,TIME_STEP);
    }

	Clear(BLACK);

    RLNeuralNetworkEnv* shown = nt.rlnnenvs + ai;
    PongGame_Render(shown->env);

    String str = String_Format("Training: %d, Ai: %d, F:%f, AF:%f",training,ai,shown->rlnn.nenv.fittness,shown->rlnn.nenv.avfittness);
    RenderCStrSizeAlxFont(&font,str.Memory,str.size,GetWidth() - 600.0f,0.0f,WHITE);
    String_Free(&str);

    Thread_Sleep_M(NN_DELAY);
}
void Delete(AlxWindow* w){
	for(int i = 0;i<nt.gensize;i++){
        RLNeuralNetworkEnv* rlnn = nt.rlnnenvs + i;
        
        CStr filename = CStr_Format(NN_PATH "%d." NN_PATHTYPE,i);
        NeuralNetwork_Save(&rlnn->rlnn.nn,filename);
        CStr_Free(&filename);
        
        printf("[NeuralNetwork]: Save %d -> Success!\n",i);
    }
    
    Neat_Free(&nt);
    AlxFont_Free(&font);
}

int main(){
    if(Create("Pong with AI",1920,1080,1,1,Setup,Update,Delete))
        Start();
    return 0;
}