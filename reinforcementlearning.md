#####所用模块和对应的教程
![](微信截图_20211020091235.png)
#####代替Tkinter可用
![](微信截图_20211020091716.png)
#####OpenAI gym只能在*linux*系统下运行
##### 决策过程
![](微信截图_20211020092142.png)
saxasxaxa![](微信截图_20211020092250.png)
##### Qlearning算法
![](微信截图_20211020093910.png)
![](微信截图_20211020094856.png)
#####模型建立：
1.卡尔曼滤波模型的学习：
##### 建立刺激和奖励的链接时学习的基础。有两种想法对于我们理解链接学习有深远的影响：
+ 贝叶斯学习理论说明人们不仅仅学习了联结的强度，还学习了不确定性
+ 强化学习强调了人们不仅仅受到当前刺激的强化，他们还受到未来可能存在的奖赏的影响。
#### 联结学习和R-W 模型
最具影响力的经典条件反射模型是Rescorla-Wagner模型(R-W模型）。该模型描述了一个线性系统,对奖励的预测是观测刺激和权重的线性组合
![](MommyTalk1634988946250.jpg)
而预测误差![](微信截图_20211023195452.png)反向传播更新了权重，
![](微信截图_20211023195536.png)
R-W 模型的代码复现：
```python
class RW:

    def __init__( self, dim, params=[]):
        self.dim = dim 
        self._init_params(params)
        self._init_weights()
    
    def _init_params( self, params):
        if len(params):
            self.lr = params['lr']

    def _init_weights( self):
        self.W = np.zeros( [self.dim, 1])

    def step( self, xt, rt):
        '''
        1. predict reward:
            v =  ∑_d xt(1,d)w(d,1)
        2. calculate prediction error:
            δ = rt - v
        3. update params: 
            w += α * δ * xt.T
        '''
        v   = xt @ self.W              # 1d x d1 = 1
        rpe = rt - v                   # 1
        self.W += self.lr * rpe * xt.T # d1 += 1 x 1 x d1 
```
