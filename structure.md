# <center> 基于机器视觉技术的鱼饵投喂控制算法研究 </center>
# <center> 基于机器视觉的鱼类摄食行为模型研究？（修改）</center>
## 目录
1.绪论

    1.1. 研究背景与意义

    1.2. 国内外相关研究(绪论的范围不要太宽)

        1.2.1 基于机器视觉的鱼类摄食行为模型研究现状
        1.2.2 视频分类与数据集研究现状
        1.2.3 课题组研究现状

    1.3. 本论文的主要研究工作和组织结构

        1.3.1 主要研究工作
        1.3.2 论文组织结构

    1.4. 本章小结>>>>>>>>>>>>>>>>>>>>>>>>>>0-10(10) (2019/12/22)

2.鱼类摄食行为视频采集系统与数据集构建

    2.1. 实验基地介绍

    2.2. 数据采集系统设计方案

    2.3. 大西洋鲑鱼摄食运动行为水下视频数据库

        2.3.1. 视频样本分类
        2.3.2. 视频标注系统
        2.3.3. 数据库特点分析
        2.3.4. CNN baseline

    2.4. 本章小结>>>>>>>>>>>>>>>>>>>>>>>>>>11-31(20) (2020/01/5)(14)

3.基于变分贝叶斯的摄食行为特征提取

    3.1. 图像预处理

    3.2. 变分贝叶斯推断与变分自编码器
        3.2.1 贝叶斯推断
        3.2.2 变分自编码器
        （1）MINIST 实验
         (2) UVDASSB 实验

    3.3. UVDASSB空间变换

    3.4. 本章小结>>>>>>>>>>>>>>>>>>>>>>>>>>31-46(15) (2020/01/12)

4.鱼类摄食行为分类模型

    4.1. 基于帧间关系的视频分类（视频背景建模）(MDT) [讲清楚算法的思考过程]（5）

    4.2. 基于帧间关系贝叶斯估计的鱼类行为视频分类（5）

    4.3. 基于帧间关系贝叶斯估计的鱼类行为视频分类评估（如果必要可以试试在开源的数据集上做实验论证一下）（5）

    4.4. 本章小结>>>>>>>>>>>>>>>>>>>>>>>>>>46-66(20)(2020/03/01)

5.总结与展望>>>>>>>>>>>>>>>>>>>>>>>>>>66-67(1)

