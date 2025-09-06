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
#define NN_PATH             "./data/Model.nnalx"


typedef struct PongObject{
    Vec2 p;
    Vec2 d;
    Vec2 v;
    Pixel c;
} PongObject;

PongObject paddle1;
PongObject paddle2;
PongObject ball;
int score1 = 0;
int score2 = 0;
NeuralReward aireward = 0.0f;

void Reset(){
    paddle1.p.y = 0.5f - PADDLE_HEIGHT * 0.5f;
    paddle2.p.y = 0.5f - PADDLE_HEIGHT * 0.5f;
    
    ball.p.x = 0.5f - BALL_WIDTH * 0.5f;
    ball.p.y = 0.5f - BALL_HEIGHT * 0.5f;
    //ball.v.x = BALL_STARTSPEED;
    //ball.v.y = 0.0f;

    float a = Random_f64_MinMax(0.0f,2.0f * F32_PI);
    a = F32_Border(a,0.25f * F32_PI,0.75f * F32_PI);
    a = F32_Border(a,1.25f * F32_PI,1.75f * F32_PI);

    ball.v = Vec2_Mulf(Vec2_OfAngle(a),BALL_STARTSPEED);
}

PongObject PongObject_New(Vec2 p,Vec2 d,Vec2 v,Pixel c){
    PongObject po;
    po.p = p;
    po.d = d;
    po.v = v;
    po.c = c;
    return po;
}
void PongObject_Update(PongObject* po,float t){
    po->p = Vec2_Add(po->p,Vec2_Mulf(po->v,t));

    if(po->p.x < -po->d.x){
        score2++;
        Reset();
    }
    if(po->p.x > 1.0f){
        score1++;
        Reset();
    }
    if(po->p.y < 0.0f){
        po->p.y = 0.0f;
        po->v.y *= -1.0f;
    }
    if(po->p.y > 1.0f - po->d.y){
        po->p.y = 1.0f - po->d.y;
        po->v.y *= -1.0f;
    }
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

void PongObject_Step_Update(PongObject* po,float t){
    po->p = Vec2_Add(po->p,Vec2_Mulf(po->v,t));

    if(po->p.x < -po->d.x){
        aireward += 10.0f;
    }
    if(po->p.x > 1.0f){
        aireward -= 1.0f;
    }
    if(po->p.y < 0.0f){
        po->p.y = 0.0f;
        po->v.y *= -1.0f;
    }
    if(po->p.y > 1.0f - po->d.y){
        po->p.y = 1.0f - po->d.y;
        po->v.y *= -1.0f;
    }
}
void PongObject_Step_Collision(PongObject* po,PongObject* target){
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

        if(po == &paddle2){
            aireward += 1.0f;
        }
    }
}



#define NN_INPUTS       4
#define NN_OUTPUTS      3

typedef struct PongInfos{
    NeuralType balldx;
    NeuralType bally;
    NeuralType paddle2y;
    NeuralType ballvy;
    // other step values
    NeuralType ballx;
    NeuralType paddle1y;
    NeuralType ballvx;
} PongInfos;

PongInfos PongInfos_New(){
    PongInfos pi;
    pi.balldx = F32_Abs(ball.p.x - paddle2.p.x);
    pi.bally = ball.p.y + ball.d.y * 0.5f;
    pi.paddle2y = paddle2.p.y + paddle2.d.y * 0.5f;
    pi.ballvy = ball.v.y;
    // ----------------------
    pi.ballx = ball.p.x;
    pi.paddle1y = paddle1.p.y;
    pi.ballvx = ball.v.x;
    return pi;
}


#define TIME_STEP   0.10f

char training = 0;
char ai = 0;
float ElapTime = 0.05f;
RLNeuralNetwork nn;
AlxFont font;

void NeuralEnviroment_Func_State(RLNeuralNetwork* nn,DecisionState* ds){
    PongInfos pi_before = PongInfos_New();
    DecisionState_SetBefore(ds,(NeuralType*)&pi_before);
}
void NeuralEnviroment_Func_Step(RLNeuralNetwork* nn,DecisionState* ds,int d){
    PongInfos pi_before = PongInfos_New();
    DecisionState_SetBefore(ds,(NeuralType*)&pi_before);

    NeuralReward reward = aireward;
    aireward = 0.0f;

    if(d==0)      paddle2.v.y = -PADDLE_SPEED;
    else if(d==1) paddle2.v.y = PADDLE_SPEED;
    else if(d==2) paddle2.v.y = 0.0f;
    
    PongObject_Step_Update(&paddle1,TIME_STEP);
    PongObject_Step_Update(&paddle2,TIME_STEP);
    PongObject_Step_Update(&ball,TIME_STEP);

    PongObject_Step_Collision(&paddle1,&ball);
    PongObject_Step_Collision(&paddle2,&ball);

    // aireward += (1.0f - 
    //     //Vec2_Mag(Vec2_Sub(
    //     //    Vec2_Add(paddle2.p, Vec2_Mulf(paddle2.d,0.5f)),
    //     //    Vec2_Add(ball.p,    Vec2_Mulf(ball.d,   0.5f))
    //     //
    //     (paddle2.p.y  + paddle2.d.y   * 0.5f) -
    //     (ball.p.y     + ball.d.y      * 0.5f)
    // ) * 0.5f * window.ElapsedTime;

    const Vec2 pm = Vec2_Add(paddle2.p, Vec2_Mulf(paddle2.d,0.5f)); 
    const Vec2 bm = Vec2_Add(ball.p,    Vec2_Mulf(ball.d,   0.5f)); 
    const Vec2 dp = Vec2_Sub(pm,bm);
    if(F32_Sign(dp.y) == F32_Sign(ball.v.y))
        aireward += (1.0f - dp.y) * 0.5f * window.ElapsedTime;

    NeuralReward ret = aireward;
    aireward = reward;

    PongInfos pi_after = PongInfos_New();
    DecisionState_SetAfter(ds,(NeuralType*)&pi_after);

    DecisionState_SetReward(ds,ret);
}
void NeuralEnviroment_Func_Undo(RLNeuralNetwork* nn,DecisionState* ds){
    PongInfos* pi = (PongInfos*)ds->before;
    ball.p.y = pi->bally - ball.d.y * 0.5f;
    paddle2.p.y = pi->paddle2y - paddle2.d.y * 0.5f;
    ball.v.y = pi->ballvy;
    // ----------------------
    ball.p.x = pi->ballx;
    paddle1.p.y = pi->paddle1y;
    ball.v.x = pi->ballvx;
}

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

    paddle1 = PongObject_New((Vec2){ PADDLE_1_POSX,0.0f },(Vec2){ PADDLE_WIDTH,PADDLE_HEIGHT },(Vec2){ 0.0f,0.0f },PADDLE_1_COLOR);
    paddle2 = PongObject_New((Vec2){ PADDLE_2_POSX,0.0f },(Vec2){ PADDLE_WIDTH,PADDLE_HEIGHT },(Vec2){ 0.0f,0.0f },PADDLE_2_COLOR);
    ball = PongObject_New((Vec2){ 0.0f,0.0f },(Vec2){ BALL_WIDTH,BALL_HEIGHT },(Vec2){ 0.0f,0.0f },BALL_COLOR);
    Reset();

    nn = RLNeuralNetwork_New(
        NeuralNetwork_Make((NeuralLayerBuilder[]){
            NeuralLayerBuilder_Make(NN_INPUTS,"relu"),
            NeuralLayerBuilder_Make(32,"relu"),
            NeuralLayerBuilder_Make(NN_OUTPUTS,"softmax"),
            NeuralLayerBuilder_End()
        }),
        NeuralEnviroment_New(
            (void*)NeuralEnviroment_Func_State,
            (void*)NeuralEnviroment_Func_Step,
            (void*)NeuralEnviroment_Func_Undo,
            sizeof(PongInfos) / sizeof(NeuralType),
            NN_OUTPUTS
        )
    );
}
void Update(AlxWindow* w){
    ReloadAlxFont(GetWidth() * 0.025f,GetHeight() * 0.05f);

    if(Stroke(ALX_KEY_T).PRESSED){
        training = !training;
    }
    if(Stroke(ALX_KEY_Z).PRESSED){
        ai = ai + 1;
        if(ai > 3) ai = 0;
    }
    if(Stroke(ALX_KEY_E).PRESSED){
        NeuralNetwork_Save(&nn.nn,NN_PATH);
        printf("[NeuralNetwork]: Save -> Success!\n");
    }
    if(Stroke(ALX_KEY_D).PRESSED){
        NeuralNetwork_Free(&nn.nn);
        if(Files_isFile(NN_PATH)){
            nn.nn = NeuralNetwork_Load(NN_PATH);
            printf("[NeuralNetwork]: Load -> Success!\n");
        }else{
            nn = RLNeuralNetwork_New(
                NeuralNetwork_Make((NeuralLayerBuilder[]){
                    NeuralLayerBuilder_Make(NN_INPUTS,"relu"),
                    NeuralLayerBuilder_Make(32,"relu"),
                    NeuralLayerBuilder_Make(NN_OUTPUTS,"softmax"),
                    NeuralLayerBuilder_End()
                }),
                NeuralEnviroment_New(
                    (void*)NeuralEnviroment_Func_State,
                    (void*)NeuralEnviroment_Func_Step,
                    (void*)NeuralEnviroment_Func_Undo,
                    sizeof(PongInfos) / sizeof(NeuralType),
                    NN_OUTPUTS
                )
            );
            printf("[NeuralNetwork]: Load -> Failed!\n");
        }
    }

    w->ElapsedTime = ElapTime;
    if(Stroke(ALX_KEY_Q).DOWN)  ElapTime *= 1.01f;
    if(Stroke(ALX_KEY_A).DOWN)  ElapTime *= 0.99f;

    //if(Stroke(ALX_KEY_W).DOWN)          paddle1.v.y = -PADDLE_SPEED;
    //else if(Stroke(ALX_KEY_S).DOWN)     paddle1.v.y = PADDLE_SPEED;
    //else                                paddle1.v.y = 0.0f;

    //if(Stroke(ALX_KEY_UP).DOWN)         paddle2.v.y = -PADDLE_SPEED;
    //else if(Stroke(ALX_KEY_DOWN).DOWN)  paddle2.v.y = PADDLE_SPEED;
    //else                                paddle2.v.y = 0.0f;


    if(training){
        RLNeuralNetwork_LearnDecisionState(&nn,NULL,NN_DELTA,NN_LEARNRATE);
    }

    const float dy = (ball.p.y + ball.d.y * 0.5f) - (paddle1.p.y + paddle1.d.y * 0.5f + paddle1.d.y * 0.2f * Random_f64_MinMax(-1.0f,1.0f));
    paddle1.v.y = PADDLE_SPEED * F32_Sign(dy);

    if(ai == 0){
        PongInfos pi = PongInfos_New();
        NeuralNetwork_Calc(&nn.nn,(NeuralType*)&pi);

        int d = NeuralNetwork_Decision(&nn.nn);
        if(d==0)      paddle2.v.y = -PADDLE_SPEED;
        else if(d==1) paddle2.v.y = PADDLE_SPEED;
        else if(d==2) paddle2.v.y = 0.0f;
        else          printf("Error!\n");
    }else if(ai == 1){
        int state = (int)Random_u32_MinMax(0,3) - 1;
        paddle2.v.y = PADDLE_SPEED * state;
    }else if(ai == 2){
        const float dy = (ball.p.y + ball.d.y * 0.5f) - (paddle2.p.y + paddle2.d.y * 0.5f + paddle1.d.y * 0.5f * Random_f64_MinMax(-1.0f,1.0f));
        paddle2.v.y = PADDLE_SPEED * F32_Sign(dy);
    }else{
        if(Stroke(ALX_KEY_UP).DOWN)         paddle2.v.y = -PADDLE_SPEED;
        else if(Stroke(ALX_KEY_DOWN).DOWN)  paddle2.v.y = PADDLE_SPEED;
        else                                paddle2.v.y = 0.0f;
    }

    PongObject_Update(&paddle1,w->ElapsedTime);
    PongObject_Update(&paddle2,w->ElapsedTime);
    PongObject_Update(&ball,w->ElapsedTime);
    
    PongObject_Collision(&paddle1,&ball);
    PongObject_Collision(&paddle2,&ball);

	Clear(BLACK);

    PongObject_Render(&paddle1);
    PongObject_Render(&paddle2);
    PongObject_Render(&ball);

    NeuralNetwork_Render(&nn.nn);

    String str = String_Format("Training: %d, Ai: %d, T:%f, R:%f",training,ai,ElapTime,aireward);
    RenderCStrSizeAlxFont(&font,str.Memory,str.size,GetWidth() - 600.0f,0.0f,WHITE);
    String_Free(&str);

    str = String_Format("%d : %d",score1,score2);
    RenderCStrSize(str.Memory,str.size,(GetWidth() - str.size * GetAlxFont()->CharSizeX) * 0.5f,0.0f,WHITE);
    String_Free(&str);

    Thread_Sleep_M(NN_DELAY);
}
void Delete(AlxWindow* w){
	RLNeuralNetwork_Free(&nn);
    AlxFont_Free(&font);
}

int main(){
    if(Create("Pong with AI",1920,1080,1,1,Setup,Update,Delete))
        Start();
    return 0;
}