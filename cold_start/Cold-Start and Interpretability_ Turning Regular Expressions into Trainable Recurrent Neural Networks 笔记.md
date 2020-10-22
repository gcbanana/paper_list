# Cold-Start and Interpretability: Turning Regular Expressions into Trainable Recurrent Neural Networks 笔记

emnlp2020
[http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/emnlp20reg.pdf](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/emnlp20reg.pdf)
[https://github.com/jeffchy/RE2RNN](https://github.com/jeffchy/RE2RNN)
# Idea
分类模型冷启动（有部分模板，没有标注好的训练数据），后续持续训练（有标注训练数据优化参数），将正则表达式转换的 DFA/WFA 融合成 RNN 的 recurrent 表达形式，可解释性（后续优化后的 RNN 可将参数矩阵重新转化回 DFA/WFA）
# Background
分类问题冷启动，工业界一般都是上正则，而正则表达式可以转换为有限状态自动机，如图：
![](https://github.com/gcbanana/paper_list/blob/main/pic/FARNN_1.jpg)
正则匹配后还有一个逻辑层才能得到标签，例如：![](https://cdn.nlark.com/yuque/__latex/f56deeeb60f32161e65110eaeb0eaad3.svg#card=math&code=%28M_%7Bi%7D%5Cvee%20M_%7Bj%7D%29%5Cwedge%20%5Cneg%20M_%7Bk%7D%5CRightarrow%20l_%7Bp%7D&height=21&width=169)，![](https://cdn.nlark.com/yuque/__latex/e9728c4330012bb285d12f5b50f4621b.svg#card=math&code=M_%7Bi%7D&height=18&width=21) 表示匹配上了第 i 个正则表达式
### WFA
论文除了提到 m-DFA，还提到了 weighted finite-state automaton (WFA)，它和 DFA 的区别就是给每个 transition 加了一个 weight，而正常的 DFA 每两个状态之间的 transition 是 1 或 0，WFA 由五项构成：![](https://cdn.nlark.com/yuque/__latex/dca0ec54cab829b6b56478296ab3ef9c.svg#card=math&code=A%3D%3C%5CSigma%20%2CS%2CT%2C%5Calpha_%7B0%7D%2C%5Calpha_%7B%5Cinfty%7D%3E&height=18&width=173)

- ![](https://cdn.nlark.com/yuque/__latex/025b3f94d79319f2067156076bf05243.svg#card=math&code=%5CSigma%20&height=16&width=12)：有限的输入词典，![](https://cdn.nlark.com/yuque/__latex/9054fd14ad01a319248fab8c4116bac2.svg#card=math&code=%7C%5CSigma%20%7C%3DV&height=20&width=57)
- ![](https://cdn.nlark.com/yuque/__latex/5dbc98dcc983a70728bd082d1a47546e.svg#card=math&code=S&height=16&width=10)：有限的状态集合，![](https://cdn.nlark.com/yuque/__latex/80ab7ae9196599d607dac9411bc99c79.svg#card=math&code=%7CS%7C%3DK&height=20&width=57)
- ![](https://cdn.nlark.com/yuque/__latex/cb80e8600ac6bb9ec49ad300366749bd.svg#card=math&code=T%20%5Cin%20R%5E%7BV%20%5Ctimes%20K%20%5Ctimes%20K%7D&height=19&width=95)：三维张量，![](https://cdn.nlark.com/yuque/__latex/c07bf4551aac8e2767851f771527824d.svg#card=math&code=T%5B%5Csigma%2Ci%2Cj%5D&height=20&width=58) 表示当前 ![](https://cdn.nlark.com/yuque/__latex/e5a7472d780a5a032c7775cc5e3ce901.svg#card=math&code=s_%7Bi%7D&height=14&width=13) 状态，输入进了词 ![](https://cdn.nlark.com/yuque/__latex/a2ab7d71a0f07f388ff823293c147d21.svg#card=math&code=%5Csigma&height=12&width=9)，能够达到 ![](https://cdn.nlark.com/yuque/__latex/3544084e4e8619625a93b6de1bae2ced.svg#card=math&code=s_%7Bj%7D&height=16&width=14) 状态的 transition weight，这个 weight 可以是 0 或 1，也可以是其他数，![](https://cdn.nlark.com/yuque/__latex/11f0a2270dd6bfe261bdc889c85a4e5d.svg#card=math&code=T%5B%5Csigma%5D%5Cin%20R%5E%7BK%5Ctimes%20K%7D%20&height=23&width=96) 即为 ![](https://cdn.nlark.com/yuque/__latex/a2ab7d71a0f07f388ff823293c147d21.svg#card=math&code=%5Csigma&height=12&width=9) 的转移矩阵
- ![](https://cdn.nlark.com/yuque/__latex/47e4a04f2b56a94ea55056e38696c311.svg#card=math&code=%5Calpha_%7B0%7D%5Cin%20R%5E%7BK%7D&height=21&width=64)：初始 weights，![](https://cdn.nlark.com/yuque/__latex/ec106d8c7914c8ec2956cc39338e85ab.svg#card=math&code=%5Calpha_%7B0%7D%5Bi%5D%3D%5Cmathbb%20I%5C%7Bs_%7Bi%7D%5Cin%20S_%7B0%7D%5C%7D&height=20&width=131)，![](https://cdn.nlark.com/yuque/__latex/88c53eb48ff7177709b77e54d00da0b9.svg#card=math&code=%5Cmathbb%20I&height=16&width=6) 是指示函数；FA 只有一个初始状态，但具有多个初始状态的 FA 可以转换为有一个初始状态的 FA，就是在前面加一个新的 state 节点然后发射到多个初始状态即可
- ![](https://cdn.nlark.com/yuque/__latex/c455d627c6f72488b85ecd1a42d8edc9.svg#card=math&code=%5Calpha_%7B%5Cinfty%7D%5Cin%20R%5E%7BK%7D&height=21&width=70)：结束 weights，![](https://cdn.nlark.com/yuque/__latex/052cdfd0e7a04f917d9e9cc5a0829430.svg#card=math&code=%5Calpha_%7B%5Cinfty%7D%5Bi%5D%3D%5Cmathbb%20I%5C%7Bs_%7Bi%7D%5Cin%20S_%7B%5Cinfty%7D%5C%7D&height=20&width=143)
### Forward Algorithm
论文除了前向也提到了维特比，但维特比效果不如前向
x 经过 state path ![](https://cdn.nlark.com/yuque/__latex/59b9d0fb350173576e0bf1d5e2c6cadc.svg#card=math&code=p%20%3D%20%3Cu_%7B1%7D%2C...%2Cu_%7BN%2B1%7D%3E&height=16&width=154)，path 的得分为 ![](https://cdn.nlark.com/yuque/__latex/160e82760c55186200fb2f0347c9e445.svg#card=math&code=%5Calpha_%7B0%7D%5Bu_%7B1%7D%5D%5Ccdot%28%5Cprod_%7Bi%3D1%7D%5E%7BN%7DT%5Bx_%7Bi%7D%2Cu_%7Bi%7D%2Cu_%7Bi%2B1%7D%5D%29%5Ccdot%20%5Calpha_%7B%5Cinfty%7D%5Bu_%7BN%2B1%7D%5D&height=53&width=275)
![](https://cdn.nlark.com/yuque/__latex/7df069b49699bf0b511a70c3820c82f2.svg#card=math&code=B_%7Bforward%7D%28A%2Cx%29%3D%5Csum_%7Bp%5Cin%20%5Cpi%28x%29%7DB%28A%2Cp%29%3D%5Calpha_%7B0%7D%5E%7BT%7D%5Ccdot%28%5Cprod_%7Bi%3D1%7D%5E%7BN%7DT%5Bx_%7Bi%7D%5D%29%5Ccdot%20%5Calpha_%7B%5Cinfty%7D&height=57&width=383)
# Method
前向过程其实可以看成是 RNN 的一个 recurrent 的形式；把 ![](https://cdn.nlark.com/yuque/__latex/23986d9b2a622a418cb84ad397a8df80.svg#card=math&code=h_%7Bt%7D&height=18&width=15) 看成是经过了 t 个 words 后的 score 向量，维度和自动机状态数相同；![](https://cdn.nlark.com/yuque/__latex/a53f406192698a5da80ee69b39e9be71.svg#card=math&code=h_%7Bt%7D%5Bi%5D&height=20&width=30) 看成是从 ![](https://cdn.nlark.com/yuque/__latex/6d155a8ec86cc6633458655c91f23d08.svg#card=math&code=s_%7B0%7D&height=14&width=15) 开始到 ![](https://cdn.nlark.com/yuque/__latex/e5a7472d780a5a032c7775cc5e3ce901.svg#card=math&code=s_%7Bi%7D&height=14&width=13) 的路径的数量，即：
![](https://cdn.nlark.com/yuque/__latex/5f6ec62e9f83a576879e3df37fb56d56.svg#card=math&code=h_%7B0%7D%3D%5Calpha_%7B0%7D%5E%7BT%7D&height=23&width=60)
![](https://cdn.nlark.com/yuque/__latex/c4468a8b214d02583608b821f7294d1d.svg#card=math&code=h_%7Bt%7D%3Dh_%7Bt-1%7D%5Ccdot%20T%5Bx_t%5D%2C1%3C%3Dt%3C%3DN&height=20&width=226)
![](https://cdn.nlark.com/yuque/__latex/38108b6a0b58a747217ecf499b4113ed.svg#card=math&code=B_%7Bforward%7D%28A%2Cx%29%3Dh_%7BN%7D%5Ccdot%20%5Calpha_%7B%5Cinfty%7D&height=21&width=182)
至此，WFA 转换为了一个 RNN，参数为 ![](https://cdn.nlark.com/yuque/__latex/f33db9e4f9eaa973326c5815ea7756a8.svg#card=math&code=%5CTheta%3D%3C%5Calpha_%7B0%7D%2CT%2C%5Calpha_%7B%5Cinfty%7D%3E&height=18&width=135)
## Decomposing
T 是一个三维 tensor，参数量大，为了降低参数量，用了张量 CP 分解来做近似
### CPD
![](https://github.com/gcbanana/paper_list/blob/main/pic/FARNN_2.jpg)
设三维 tensor ![](https://cdn.nlark.com/yuque/__latex/2e8aa69866123de86a2ecc90f88c9281.svg#card=math&code=T%20%5Cin%20R%5E%7Bd_%7B1%7D%5Ctimes%20d_%7B2%7D%5Ctimes%20d_%7B3%7D%7D&height=19&width=102)，可近似为：
![](https://cdn.nlark.com/yuque/__latex/fe0a4155e69e3c4d7a1c321a04fc98ff.svg#card=math&code=T%5Capprox%20%5Chat%7BT%7D%3D%5Csum_%7Bi%3D1%7D%5E%7Br%7Da_%7Bi%7D%5Cotimes%20b_%7Bi%7D%5Cotimes%20c_%7Bi%7D&height=49&width=178)
![](https://cdn.nlark.com/yuque/__latex/50b40f766fb4824a555728b4e972860f.svg#card=math&code=a_%7Bi%7D%5Cin%20R%5E%7Bd_%7B1%7D%7D&height=21&width=61)，![](https://cdn.nlark.com/yuque/__latex/9b89ec8180e67594cc6d9c6d7de7e7e1.svg#card=math&code=b_%7Bi%7D%5Cin%20R%5E%7Bd_%7B2%7D%7D&height=21&width=60)，![](https://cdn.nlark.com/yuque/__latex/37f074a248c144a84d515cccaa3a7f1d.svg#card=math&code=c_%7Bi%7D%5Cin%20R%5E%7Bd_%7B3%7D%7D&height=21&width=60)，![](https://cdn.nlark.com/yuque/__latex/790c76ceb13e928d08edc53d7ac4bb5c.svg#card=math&code=%5Cotimes&height=16&width=12) 是 outer product，即，
![](https://cdn.nlark.com/yuque/__latex/5578709babc6a3590bb34c7bfce667dd.svg#card=math&code=T_%7Bijk%7D%5Capprox%20%5Csum_%7Bi%3D1%7D%5E%7Br%7Da_%7Bri%7Db_%7Brj%7Dc_%7Brk%7D&height=49&width=135)
另 ![](https://cdn.nlark.com/yuque/__latex/e59c6dc0fc616fac895dd52867165859.svg#card=math&code=%5Chat%7BT%7D_%7B%281%29%7D&height=26&width=29) 是 ![](https://cdn.nlark.com/yuque/__latex/177489ca0526a65cfabd1ee858229ede.svg#card=math&code=%5Chat%7BT%7D&height=20&width=12) 的 model-1 unfolding，则 ![](https://cdn.nlark.com/yuque/__latex/f7c96aa37426a9d40cc18225e2e9d56f.svg#card=math&code=%5Chat%7BT%7D_%7B%281%29%7D%5Cin%20R%5E%7Bd_%7B1%7D%5Ctimes%20%28d_%7B2%7Dd_%7B3%7D%29%7D&height=26&width=119)，论文好像这里写错，应该是 ![](https://cdn.nlark.com/yuque/__latex/6138f7825d75a5bfbd9cb452888ca212.svg#card=math&code=%5Chat%7BT%7D_%7B%281%29%7D%3DA%28C%5Codot%20B%29%5E%7BT%7D&height=26&width=134) 而不是 ![](https://cdn.nlark.com/yuque/__latex/ee4ee467568910d586d4561cdd339d01.svg#card=math&code=%5Chat%7BT%7D_%7B%281%29%7D%3D%28C%5Codot%20B%29A%5E%7BT%7D&height=26&width=134)
![](https://cdn.nlark.com/yuque/__latex/319d584a4a5166ee6c51f4b8348856ea.svg#card=math&code=%5Codot&height=16&width=12) 是 Khatri-Rao product
![](https://cdn.nlark.com/yuque/__latex/f02ce8e2118ff7437b4bdbbd104c1363.svg#card=math&code=A%3D%5Ba_%7B1%7D...a_%7Br%7D%5D%2CA%5Cin%20R%5E%7Bd_%7B1%7D%5Ctimes%20r%7D&height=23&width=181)
![](https://cdn.nlark.com/yuque/__latex/104650e5c8eb50afc36631c6fb74d760.svg#card=math&code=B%3D%5Bb_%7B1%7D...b_%7Br%7D%5D%2CB%5Cin%20R%5E%7Bd_%7B2%7D%5Ctimes%20r%7D&height=23&width=178)
![](https://cdn.nlark.com/yuque/__latex/4c0536c7dac6e5d4409fbbf4a5517db5.svg#card=math&code=C%3D%5Bc_%7B1%7D...c_%7Br%7D%5D%2CC%5Cin%20R%5E%7Bd_%7B3%7D%5Ctimes%20r%7D&height=23&width=178)
#### Outer Product
设 d1=4，d2=3，d3=2，r=1，有：
![](https://cdn.nlark.com/yuque/__latex/08665219c45e9b86cd3aac523c922c84.svg#card=math&code=a_%7B1%7D%5Cotimes%20b_%7B1%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%0Aa_%7B11%7D%5C%5C%20%0Aa_%7B12%7D%5C%5C%20%0Aa_%7B13%7D%5C%5C%20%0Aa_%7B14%7D%0A%5Cend%7Bbmatrix%7D%20%5Cotimes%20%5Cbegin%7Bbmatrix%7D%0Ab_%7B11%7D%5C%5C%20%0Ab_%7B12%7D%5C%5C%20%0Ab_%7B13%7D%0A%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%0Aa_%7B11%7Db_%7B11%7D%20%26%20a_%7B11%7Db_%7B12%7D%20%26%20a_%7B11%7Db_%7B13%7D%5C%5C%20%0Aa_%7B12%7Db_%7B11%7D%20%26%20a_%7B12%7Db_%7B12%7D%20%26%20a_%7B12%7Db_%7B13%7D%5C%5C%20%0Aa_%7B13%7Db_%7B11%7D%20%26%20a_%7B13%7Db_%7B12%7D%20%26%20a_%7B13%7Db_%7B13%7D%5C%5C%20%0Aa_%7B14%7Db_%7B11%7D%20%26%20a_%7B14%7Db_%7B12%7D%20%26%20a_%7B14%7Db_%7B13%7D%0A%5Cend%7Bbmatrix%7D&height=90&width=407)
![](https://cdn.nlark.com/yuque/__latex/8c167a894ae7a6a8bdee6c63afe09feb.svg#card=math&code=X%20%3D%20a_%7B1%7D%5Cotimes%20b_%7B1%7D%5Cotimes%20c_%7B1%7D%3D%5Cbegin%7Bbmatrix%7D%0Aa_%7B11%7D%5C%5C%20%0Aa_%7B12%7D%5C%5C%20%0Aa_%7B13%7D%5C%5C%20%0Aa_%7B14%7D%0A%5Cend%7Bbmatrix%7D%20%5Cotimes%20%5Cbegin%7Bbmatrix%7D%0Ab_%7B11%7D%5C%5C%20%0Ab_%7B12%7D%5C%5C%20%0Ab_%7B13%7D%0A%5Cend%7Bbmatrix%7D%5Cotimes%20%5Cbegin%7Bbmatrix%7D%0Ac_%7B11%7D%5C%5C%20%20%0Ac_%7B12%7D%0A%5Cend%7Bbmatrix%7D&height=90&width=330)
则有：
![](https://cdn.nlark.com/yuque/__latex/7a8471c5ae8db7f1628e77591136c12a.svg#card=math&code=X%28%3A%2C%3A%2C1%29%3D%5Cbegin%7Bbmatrix%7D%0Aa_%7B11%7Db_%7B11%7Dc_%7B11%7D%20%26%20a_%7B11%7Db_%7B12%7Dc_%7B11%7D%20%26%20a_%7B11%7Db_%7B13%7Dc_%7B11%7D%5C%5C%20%0Aa_%7B12%7Db_%7B11%7Dc_%7B11%7D%20%26%20a_%7B12%7Db_%7B12%7Dc_%7B11%7D%20%26%20a_%7B12%7Db_%7B13%7Dc_%7B11%7D%5C%5C%20%0Aa_%7B13%7Db_%7B11%7Dc_%7B11%7D%20%26%20a_%7B13%7Db_%7B12%7Dc_%7B11%7D%20%26%20a_%7B13%7Db_%7B13%7Dc_%7B11%7D%5C%5C%20%0Aa_%7B14%7Db_%7B11%7Dc_%7B11%7D%20%26%20a_%7B14%7Db_%7B12%7Dc_%7B11%7D%20%26%20a_%7B14%7Db_%7B13%7Dc_%7B11%7D%0A%5Cend%7Bbmatrix%7D&height=90&width=336)
![](https://cdn.nlark.com/yuque/__latex/6454ba1c3649175eef229808d19d4d6e.svg#card=math&code=X%28%3A%2C%3A%2C2%29%3D%5Cbegin%7Bbmatrix%7D%0Aa_%7B11%7Db_%7B11%7Dc_%7B12%7D%20%26%20a_%7B11%7Db_%7B12%7Dc_%7B12%7D%20%26%20a_%7B11%7Db_%7B13%7Dc_%7B12%7D%5C%5C%20%0Aa_%7B12%7Db_%7B11%7Dc_%7B12%7D%20%26%20a_%7B12%7Db_%7B12%7Dc_%7B12%7D%20%26%20a_%7B12%7Db_%7B13%7Dc_%7B12%7D%5C%5C%20%0Aa_%7B13%7Db_%7B11%7Dc_%7B12%7D%20%26%20a_%7B13%7Db_%7B12%7Dc_%7B12%7D%20%26%20a_%7B13%7Db_%7B13%7Dc_%7B12%7D%5C%5C%20%0Aa_%7B14%7Db_%7B11%7Dc_%7B12%7D%20%26%20a_%7B14%7Db_%7B12%7Dc_%7B12%7D%20%26%20a_%7B14%7Db_%7B13%7Dc_%7B12%7D%0A%5Cend%7Bbmatrix%7D&height=90&width=336)
#### Model-1 Unfolding
上面的例子中：![](https://cdn.nlark.com/yuque/__latex/f1810aebb8c8f6ea3deb4b65563a845b.svg#card=math&code=X_%7B%281%29%7D%3D%5BX%28%3A%2C%3A%2C1%29%2CX%28%3A%2C%3A%2C2%29%5D&height=23&width=190)
size 为 (4, 6)
#### Khatri-Rao Product
给定大小为 ![](https://cdn.nlark.com/yuque/__latex/2dc13003cc069b1027f12896b1a00631.svg#card=math&code=m%20%5Ctimes%20k&height=16&width=44) 的矩阵 ![](https://cdn.nlark.com/yuque/__latex/194c4fb51436b2c51fe6492b4d6cd876.svg#card=math&code=A%3D%28%5Cvec%7Ba_1%7D%2C%5Cvec%7Ba_2%7D%2C...%2C%5Cvec%7Ba_k%7D%29&height=25&width=143) 和 大小为 ![](https://cdn.nlark.com/yuque/__latex/cf569ff60f8f5310adb505ae48960119.svg#card=math&code=n%20%5Ctimes%20k&height=16&width=39) 的 ![](https://cdn.nlark.com/yuque/__latex/b9b704a8124bacbac175e247c704c634.svg#card=math&code=B%3D%28%5Cvec%7Bb_1%7D%2C%5Cvec%7Bb_2%7D%2C...%2C%5Cvec%7Bb_k%7D%29&height=30&width=144)，A 和 B 的 Khatri-Rao 积为：
![](https://cdn.nlark.com/yuque/__latex/e1b04391d5f55999881fe54b653b8516.svg#card=math&code=A%5Codot%20B%3D%28%5Cvec%7Ba_1%7D%5Cotimes%20%5Cvec%7Bb_1%7D%2C%20%5Cvec%7Ba_2%7D%5Cotimes%20%5Cvec%7Bb_2%7D%2C...%2C%5Cvec%7Ba_k%7D%5Cotimes%20%5Cvec%7Bb_k%7D%29&height=30&width=288)，注意这里的 ![](https://cdn.nlark.com/yuque/__latex/790c76ceb13e928d08edc53d7ac4bb5c.svg#card=math&code=%5Cotimes&height=16&width=12) 不是 outer product 而是 Kronecker 积


回到上面的例子，
![](https://cdn.nlark.com/yuque/__latex/968ff725c8c68ca3648989f444a48cf0.svg#card=math&code=C%5Codot%20B%3D%28%5Cvec%7Ba_1%7D%5Cotimes%20%5Cvec%7Bb_1%7D%29%3D%28%5Cbegin%7Bbmatrix%7D%0Ac_%7B11%7D%5C%5C%20%20%0Ac_%7B12%7D%0A%5Cend%7Bbmatrix%7D%5Cotimes%20%5Cbegin%7Bbmatrix%7D%0Ab_%7B11%7D%5C%5C%20%0Ab_%7B12%7D%5C%5C%20%0Ab_%7B13%7D%0A%5Cend%7Bbmatrix%7D%29%3D%5Cbegin%7Bbmatrix%7D%0Ac_%7B11%7Db_%7B11%7D%5C%5C%20%0Ac_%7B11%7Db_%7B12%7D%5C%5C%20%0Ac_%7B11%7Db_%7B13%7D%5C%5C%0Ac_%7B12%7Db_%7B11%7D%5C%5C%0Ac_%7B12%7Db_%7B12%7D%5C%5C%0Ac_%7B12%7Db_%7B13%7D%0A%5Cend%7Bbmatrix%7D&height=139&width=376)
易知
![](https://cdn.nlark.com/yuque/__latex/0a4fd3be4da330a4ea868fd86652942d.svg#card=math&code=%5Chat%7BT%7D_%7B%281%29%7D%3DA%28C%5Cbigodot%20B%29%5E%7BT%7D&height=28&width=144)


**r 小于 tensor 的 rank时，dp 分解是近似的；实验表明 100 个 state 的 FA，当 r>=100 时，decomposition error<=1%**


于是将公式中的 tensor T 分解为 ![](https://cdn.nlark.com/yuque/__latex/15acb06c735478e6cc510d40d5cc37cb.svg#card=math&code=E_R%2CD_1%2CD_2&height=18&width=81)
![](https://cdn.nlark.com/yuque/__latex/f1eaec3db27807acca743648e0df322f.svg#card=math&code=E_R%5Cin%20R%5E%7BV%5Ctimes%20r%7D&height=21&width=82)：它的第一维度和词典是相同的，因此可以看成是一个 word embedding，这个 word embedding 包含的是 RE information，对于 ![](https://cdn.nlark.com/yuque/__latex/8ac66afa5dc4aefbf764ad6d9acc8d35.svg#card=math&code=x_%7Bt%7D&height=14&width=15)，有 ![](https://cdn.nlark.com/yuque/__latex/97b73f685e2f7c90eda373859e51f3e0.svg#card=math&code=v_%7Bt%7D%5Cin%20R%5E&height=19&width=54) 作为它的词向量

之前的 recurrent 过程变换为：
![](https://cdn.nlark.com/yuque/__latex/be49ad339eefe089c6a3dffb3b0350fe.svg#card=math&code=a%3D%28h_%7Bt-1%7D%5Ccdot%20D_%7B1%7D%29%5Ccirc%20v_%7Bt%7D&height=20&width=139)
![](https://cdn.nlark.com/yuque/__latex/14757cf6c851c6f62c52ce7b7abc906f.svg#card=math&code=h_%7Bt%7D%3Da%5Ccdot%20D_%7B2%7D%5E%7BT%7D&height=23&width=82)
![](https://cdn.nlark.com/yuque/__latex/1b3c1a40f9cb094d47e8c6f9b0df773f.svg#card=math&code=%5Ccirc&height=11&width=8) 是 element-wise product

可以看出该 RNN 的 hidden state size 是 K，这个 K 可能太小，使得 RNN 的能力被限制，一个简单的做法是给 D1 和 D2 都分别 concatenate 个 ![](https://cdn.nlark.com/yuque/__latex/20c789574543e1df8cbc472014307bbc.svg#card=math&code=K%5E%7B%27%7D%20%5Ctimes%20r&height=20&width=49) 的全 0 矩阵，增加 hidden state 的 size
## Integrating Pretrained Word Embedding
上述的 RNN 只用了 CP 分解得来的含有 RE 信息的词向量矩阵，为增加语义表达又融入了一个预先训练好的 word embedding，引入一个预训练的词向量矩阵 ![](https://cdn.nlark.com/yuque/__latex/ec1d85005051b69c896e94adf1cca319.svg#card=math&code=E_%7Bw%7D%5Cin%20R%5E%7BV%5Ctimes%20D%7D&height=21&width=86)，其中 word ![](https://cdn.nlark.com/yuque/__latex/8ac66afa5dc4aefbf764ad6d9acc8d35.svg#card=math&code=x_%7Bt%7D&height=14&width=15) 的词向量为 ![](https://cdn.nlark.com/yuque/__latex/0fbea8ab29d54291bc595f052dfec43c.svg#card=math&code=u_%7Bt%7D%20%5Cin%20R%5E%7BD%7D&height=21&width=60)，又引入了一个矩阵 G，把 D 维度转换为 RE 词向量矩阵的 r 维度，初始化 G 时为了近似 RE 词向量矩阵，有 ![](https://cdn.nlark.com/yuque/__latex/a49d008bf95c8ea68cf8ca7617609c79.svg#card=math&code=G%3DE_%7Bw%7D%5E%7B%5Cdagger%7DE_%7BR%7D&height=23&width=81)，![](https://cdn.nlark.com/yuque/__latex/86a48b72d9e01902ced3cd449db2d6a6.svg#card=math&code=E_%7Bw%7D%5E%7B%5Cdagger%7D&height=23&width=22) 是 ![](https://cdn.nlark.com/yuque/__latex/88377b4bd64f633a0434ed7cba9f9527.svg#card=math&code=E_%7Bw%7D&height=18&width=22) 的伪逆

公式重写为：
![](https://cdn.nlark.com/yuque/__latex/ad0440c8dc1730fa15df8d9b11ba8f9b.svg#card=math&code=z_%7Bt%7D%3D%5Cbeta%20v_%7Bt%7D%2B%281-%5Cbeta%20%29u_%7Bt%7DG&height=20&width=160)
![](https://cdn.nlark.com/yuque/__latex/c1e0294ba68dd46c0d2caeb284526f07.svg#card=math&code=a%3D%28h_%7Bt-1%7D%5Ccdot%20D_%7B1%7D%29%5Ccirc%20z_%7Bt%7D&height=20&width=138)
![](https://cdn.nlark.com/yuque/__latex/14757cf6c851c6f62c52ce7b7abc906f.svg#card=math&code=h_%7Bt%7D%3Da%5Ccdot%20D_%7B2%7D%5E%7BT%7D&height=23&width=82)


搞了一个超参数来结合 RE 词向量和 pretrained 词向量，以上称为 FA-RNNs
## Gated Extension
牺牲了可解释性，又把 FA-RNNs 模型往 GRU 上靠拢，公式见原论文 (7)
## Bidirectional Extension
因为每个 RE 倒过来还是一条 RE，做了一个双向的 RNN，![](https://cdn.nlark.com/yuque/__latex/afd18ab51db42dad18e1cff26c9293fe.svg#card=math&code=h_%7BN%7D%3D%28%5Coverrightarrow%7Bh_%7BN%7D%7D%2B%5Coverleftarrow%7Bh_N%7D%29%2F2&height=31&width=140)
## Aggregation Layer
这一层对应着正则完了之后的分类逻辑层，相当于两层 MLP 加上 ReLU
![](https://cdn.nlark.com/yuque/__latex/00313af858385307de472b973568c581.svg#card=math&code=B_%7Bforward%7D%28A_%7Bi%7D%2Cx%29%3Dh_N%5Ccdot%20%5Cbar%7B%5Calpha%7D_%7B%5Cinfty%2Ci%7D&height=21&width=195)
![](https://cdn.nlark.com/yuque/__latex/447936ad95c0c633acffa2374b9450ef.svg#card=math&code=%5Cbar%7B%5Calpha%7D_%7B%5Cinfty%2Ci%7D&height=19&width=31) 是 ![](https://cdn.nlark.com/yuque/__latex/532872fd531309e0771a5259d7621fa5.svg#card=math&code=%5Calpha%20_%7B%5Cinfty%2Ci%7D&height=16&width=31) 在正则 ![](https://cdn.nlark.com/yuque/__latex/e8aaf87d9a5c35b14cfbc370d3fd7b21.svg#card=math&code=A_%7Bi%7D&height=18&width=18) 没有的 states 处填 0 得到的
然后是一个 soft logical expression，对应着与或非那些规则输入 label
## Training
损失函数：cross entropy
![](https://cdn.nlark.com/yuque/__latex/281b751afbe39e53cb98a406b7a8b17c.svg#card=math&code=E_%7BR%7D&height=18&width=23) 在训练中是固定住的
GloVe 词向量在训练中也固定住
## Convert Back
FA-RNN 参数有 ![](https://cdn.nlark.com/yuque/__latex/44e1adaed04163aa9f419d64be02e150.svg#card=math&code=%5CTheta_%7BRE%7D%3D%3C%5Chat%7BE%7D_%7BR%7D%2C%5Chat%7BD%7D_%7B1%7D%2C%5Chat%7BD%7D_%7B2%7D%2C%5Chat%7BG%7D%3E&height=23&width=188)，转回 WFA 就是求 tensor T，有：
![](https://cdn.nlark.com/yuque/__latex/fd89addb00d4a4bc65302a9c1d974f80.svg#card=math&code=%5Chat%7BE%7D_%7BwR%7D%3D%5Cbeta%20%5Ccdot%20%5Chat%7BE%7D_%7BR%7D%2B%281-%5Cbeta%29%5Ccdot%20E_%7Bw%7D%5Chat%7BG%7D&height=24&width=220)
![](https://cdn.nlark.com/yuque/__latex/e6ab16d36316ce423ac19bd49dd46c4e.svg#card=math&code=%5Chat%7BT%7D_%7B%281%29%7D%3D%5Chat%7BE%7D_%7BwR%7D%28%5Chat%7BD%7D_%7B2%7D%5Codot%20%5Chat%7BD%7D_%7B1%7D%29%5E%7BT%7D&height=26&width=171)
Model-1 Unfolding 这里我写的和原论文不一样


接着把 WFA 转换为 FA，使用一个阈值，![](https://cdn.nlark.com/yuque/__latex/ad96b511a8e84e9a735ebf5e25f9273f.svg#card=math&code=f%28x%29%3D%5Cmathbb%20I%5C%7Bx%5Cgeq%20%5Cgamma%5C%7D&height=20&width=119)，将 weight 转换为 0 或 1
![](https://github.com/gcbanana/paper_list/blob/main/pic/FARNN_3.jpg)
上图为原 RE 和经过学习后的 RE 对比
## Add REs
把一组新的 REs ![](https://cdn.nlark.com/yuque/__latex/3030dd246c23c32ac0ad531ba8faca08.svg#card=math&code=%5CTheta_%7Bnew%7D%3D%3CE_%7BR%7D%2CD_%7B1%7D%2CD_%7B2%7D%2CG%3E&height=18&width=191) 加进 FA-RNN
![](https://cdn.nlark.com/yuque/__latex/074b0810f856b94d745458ccad71a2e3.svg#card=math&code=%3C%5Cbegin%7Bbmatrix%7D%0A%5Chat%7BE%7D_%7BR%7D%20%26%20E_%7BR%7D%0A%5Cend%7Bbmatrix%7D%2C%5Cbegin%7Bbmatrix%7D%0A%5Chat%7BD%7D_%7B1%7D%20%26%200%5C%5C%20%0A0%20%26%20D_%7B1%7D%0A%5Cend%7Bbmatrix%7D%2C%5Cbegin%7Bbmatrix%7D%0A%5Chat%7BD%7D_%7B2%7D%20%26%200%5C%5C%20%0A0%20%26%20D_%7B2%7D%0A%5Cend%7Bbmatrix%7D%2C%5Cbegin%7Bbmatrix%7D%0A%5Chat%7BG%7D%20%26%20G%0A%5Cend%7Bbmatrix%7D%3E&height=47&width=378)
## Disable an RE
要把 WFA 中有关的 states 都删掉，再转到 FA-RNN
# Question

1. WFA 不是确定的 FA 吧
1. 假设冷启动时候写了 n 个模板，后面再怎么训练，是不是也只能学会 n 个模板已有 state 之间的有关词的转移，但不能学出新的 state，是不是冷启动模板的样式要多才能效果好呢？



