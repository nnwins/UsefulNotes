# 笔记整理

- 作者：王冰
- 类别：深度学习，自然语言处理，机器学习，编程相关，论文写作，其他
- 时间：2020年5月

[TOC]

> 时光荏苒，研究生三年的时光一晃而过。临近毕业之际，出于自我反思和自由分享的初心，整理了这么一份资料，既是对自我学习成长之路的梳理，同时也希望给学弟学妹们以借鉴，节省宝贵时间，早日入门NLP。
>

*注：以下笔记内容纯属个人观点，仅供参考，表述不当或者存在错误的地方还请多多批评指正。*

## 深度学习

- [邱锡鹏老师-神经网络与深度学习](https://nndl.github.io/)  复旦大学邱老师的书，刚出的，这些是电子版的，强烈推荐，很好的基础理论介绍的资料。
- [深度学习500问](https://github.com/scutan90/DeepLearning-500-questions) 深度学习500问，以问答形式对常用的概率知识、线性代数、**机器学习**、**深度学习**、计算机视觉等热点问题进行阐述，以帮助自己及有需要的读者。
- [某深度学习课程大纲](deeplearning/某深度学习课程大纲.pdf) 这套课程是我研一入学第一个月买的一个DL课程，算是在深度学习的一门启蒙课程吧。一般刚接触DL的人大致都需要熟悉这些方面的知识。现在想想当时真是傻，花钱买门网课，最后也没完全看完。不过外国人的上课方式就是生动有趣，分分钟让你上头。这年头很多优质的资料都是免费的，但需要你投入足够多的时间去消化。而这些入门课程，在某种程度上，对原始内容进行了某种程度降维，更加容易理解罢了。
- [某课程视频链接youtube](deeplearning/某课程视频链接youtube.pdf) 这里我当时只保存了一部分链接，感兴趣的可以看看。
- [学习率如何调节小例子](deeplearning/学习率如何调节小例子.pdf) 这是课程的一个小例子，教你学会如何根据loss来调节学习率，就是这种小例子很容易让人理解。
- [关于word_embedding](https://www.zhihu.com/question/32275069) 从one hot 到 word2vec的演变。其实后面BERT、ELmo等各种表示层出不穷。
- [关于激活函数](https://www.zhihu.com/question/22334626)  激活函数的核心：将原来只能拟合线性函数的神经网络，摇身一变，可以拟合非线性函数，这样达到可以拟合任意函数的效果了。
- [文本分类的优秀示例](https://zhuanlan.zhihu.com/p/28923961) 这是一篇知乎看山杯的比赛夺冠记录，里面将常用文本分类模型都介绍了一遍，非常值得学习。
- [详解MNIST数据集](deeplearning/详解MNIST数据集.pdf) 图像识别demo，或者CNN的demo绝对绕不开这个手写数字识别数据集，超级经典。
- [LSTM经典解读博客](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 这个是英文原版的，也可以看[中文翻译版](https://blog.csdn.net/qiu931110/article/details/69400501)。
- [RNN到LSTM最清晰易懂的视频](deeplearning/RNN到LSTM最清晰易懂的视频.pdf) 这里面提到的视频和博客是我见过关于【RNN到LSTM演变】讲的最清晰的。
- [关于BatchNormalization精华解读](deeplearning/BatchNormalization.pdf) 这是我从知乎回答里精选的内容，概括了BatchNormalization的本质。
- [各种优化器的区别](deeplearning/各种优化器的区别.pdf) 各种优化器，SGD、Adam是不是把你搞晕了，这里能解答你的疑惑。
- [Encoder-Decoder模型笔记](deeplearning/Encoder-Decoder模型笔记.pdf) 非常简洁，10分钟就能理解。
- [BiLSTM-CRF最清晰易懂的解读](https://github.com/createmomo/CRF-Layer-on-the-Top-of-BiLSTM) 这里面分了几个小部分，拆分讲解，生动形象，英文很容易懂，别怕，学会英文自由阅读就再也不怕被错误翻译给误导了有木有。
- [Transformer最佳解读-英文](https://jalammar.github.io/illustrated-transformer/) 中文版的可以参考[夏目博客](https://blog.csdn.net/qq_41664845/article/details/84969266)。还可参考[张俊林谈Attention](https://blog.csdn.net/malefactor/article/details/78767781) 和 [《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765)。
- [Transformer实战](https://wbbeyourself.github.io/2019/07/22/Transformer实战/) 这是我的博客，一行一行代码分析解读，超级详细。
- [关于Attention Model及其本质](https://blog.csdn.net/malefactor/article/details/50550211) 一文搞懂Attention的本质。
- [Seq2Seq详解](https://blog.csdn.net/Jerr__y/article/details/53749693) 揭开风靡一时的Seq2Seq的神秘面纱。
- [BERT学习](deeplearning/BERT学习.pdf) BERT学习相关链接汇总。
- [pandas常用api](deeplearning/pandas常用api.pdf)  pandas常用的api汇总。
- [关于深度学习的困惑](deeplearning/关于深度学习的困惑.pdf) 我的一些困惑，估计大家也会遇到，慢慢来，见多了就熟悉了，就会自己搞模型了。




## 自然语言处理

- [ACL Anthology](https://www.aclweb.org/anthology/)  NLPer最大的福音，一站在手，NLP论文我有。
- [AI会议投稿倒计时](https://aideadlin.es/?sub=ML,NLP,RO,SP,DM) 再也不用去各个网站看会议call for papers的截止日期了，这里全给你汇总了。
- [初学者如何查阅自然语言处理(NLP)领域学术资料](nlp/初学者如何查阅自然语言处理(NLP)领域学术资料.pdf) 关于论文查阅、NLP会议相关的资料的总结。
- [NLP应用场景](nlp/NLP应用场景.pdf) NLP技术有哪些应用呢，你一定很好奇吧？这里列举几个典型应用。
- [NLP有意思的问题汇总、典型的NLP例子](nlp/NLP有意思的问题汇总、典型的NLP例子.pdf) 非常有意思
- [NLP思维导图](nlp/NLP思维导图.pdf) NLP体系思维导图
- [NLP入门推荐](nlp/NLP入门推荐.pdf) NLP入门推荐资料，还没整理完，这只是一部分。
- [论文中出现的各种数据集缩写解释](nlp/论文中出现的各种数据集缩写解释.pdf) 现在数据集层出不穷，缩写也一大堆，搞得很晕，我这里整理一下。
- [TF-IDF与余弦相似性的应用](http://www.ruanyifeng.com/blog/2013/03/tf-idf.html) 经典有简洁的一项技术，介绍了在自动提取关键词上的应用。[代码实现](https://blog.csdn.net/liuxuejiang158blog/article/details/31360765)
- [CBOW和Skip-Gram对比](nlp/CBOW和Skip-Gram对比.pdf) 其实很简单，两句话就能说清楚。
- [神器FastText](nlp/神器FastText.pdf) 目前最高效的文本分类工具，爱了爱了。原理可以看这里[FastText原理](nlp/FastText原理.pdf)。
- [中文文本分类工具箱](https://github.com/649453932/Chinese-Text-Classification-Pytorch) 中文文本分类，TextCNN，TextRNN，**FastText**，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。业界良心啊。
- [关于模型融合方法](nlp/关于模型融合方法.pdf) 差异性越大，模型融合效果越好。
- [smp2018会议总结](nlp/smp2018会议总结.pdf) 这是我18年在哈尔滨和纪师兄参加的第一个正式学术会议，我们需要在技术评测报告环节进行赛题解决方案汇报，期间记录了一些感兴趣的报告内容。希望大家也养成随手记录的习惯，会收益颇丰。
- [使用Keras进行深度学习系列](http://www.tensorflownews.com/2018/04/12/text-cnn/)  这是一个系列，看目录就知道，我最喜欢这种系统性的教程，由浅入深，清晰全面。
- [TextCNN模型相关](nlp/TextCNN模型相关.pdf) TextCNN模型相关介绍，代码实现，原始论文解读等。
- [用深度学习解决大规模文本分类问题-综述和实践](https://zhuanlan.zhihu.com/p/25928551) 写得非常好的一篇关于文本分类的综述性文章。
- [标注偏置问题](https://blog.csdn.net/happyzhouxiaopei/article/details/7960884) CRF模型解决了标注偏置问题，去除了HMM中两个不合理的假设，当然，模型相应得也变复杂了。
- [一文搞懂HMM（隐马尔可夫模型）](https://www.cnblogs.com/skyme/p/4651331.html)  HMM是CRF的前身，值得了解。
- [论文阅读总结——Chinese NER Using Lattice LSTM](https://blog.csdn.net/qq_32728345/article/details/81264853) Lattice论文解读，可以了解一下。
- [Attention本质理解](nlp/Attention本质理解.pdf)
- [神经机器翻译(NMT)发展脉络综述](nlp/神经机器翻译(NMT)发展脉络综述.pdf) 参考2017年 神经机器翻译综述  作者：李亚超，熊德意，张民 刊物：计算机学报
- [CHIP往届比赛调研](nlp/CHIP往届比赛调研.pdf) 一个问句相似度匹配计算的任务，我当时和中科院自动化的同学一起打的，这是我对往届的调研资料。[比赛数据下载](https://www.biendata.com/user/login/?next=/competition/chip2018/data/)
- [BERT论文笔记 ](https://mp.weixin.qq.com/s/oqCRswaOhAEJ9GYZwKaKYA) 横扫各大NLP任务的BERT，你确定不看一眼吗？
- [BERT相关链接](nlp/BERT相关链接.pdf) 
- [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699) 张俊林老师写的综述，我资料里很多优质内容都出自于张老师之手，是这个领域的资深专家。
- [国内知名的自然语言处理(NLP)团队](https://www.cnblogs.com/bymo/p/8479583.html)  这篇博客发表于2018-02-27，最新的情况会有些出入，但可以参考。

## 机器学习

- [深度学习500问](https://github.com/scutan90/DeepLearning-500-questions) 深度学习500问，以问答形式对常用的概率知识、线性代数、**机器学习**、**深度学习**、计算机视觉等热点问题进行阐述，以帮助自己及有需要的读者。
- [如何解释召回率与准确率-知乎](ml/如何解释召回率与准确率-知乎.pdf) Accuracy、Precision、Recall、F1-score
- [Regularization正则化](ml/Regularization正则化.pdf) L1、L2正则化的本质
- [随机森林(Random Forest)](ml/https://blog.csdn.net/ac540101928/article/details/51689505) 随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是**决策树**，而它的本质属于机器学习的一大分支——集成学习（Ensemble Learning）方法。
- [聚类效果好坏的评价指标](ml/https://blog.csdn.net/chixujohnny/article/details/51852633)
- [ROC曲线](https://www.cnblogs.com/webRobot/p/6803747.html) ROC 曲线是根据一系列不同的二分类方式（分界值或决定阈），以真阳性率（灵敏度）为纵坐标，假阳性率（1-特异度）为横坐标绘制的曲线。
- [数据增强](ml/数据增强.pdf) 简单来说就是一种扩充数据的手段。
- [kaggle解题常规思路](ml/kaggle解题常规思路.pdf)
- [机器学习二分类问题评价指标](ml/机器学习二分类问题评价指标.pdf) 换一个角度理解二分类评价指标，发现也就那么回事。
- [ML_DL常考知识点](ml/ML_DL常考知识点.pdf) 我自己总结的，问题答案都有。

## 编程相关

这块设计内容较广，也是我们平常最经常会遇到问题的地方，不过由于知识点过于琐碎，就不列举了。而且这块的学习一般也是遇到问题，然后去搜索答案。以下我把一些理念性的东西分享一下。


- [程序设计基本原则](program/程序设计基本原则.pdf) 这是我一贯秉持的设计理念，需要用心体会。
- [不会Git不是一个合格的程序员](https://www.liaoxuefeng.com/wiki/896043488029600/896067008724000) Git不仅是个时光穿梭机，更重要的是可以看到版本迭代的点点滴滴。
- [Git Flow工作流总结](https://www.jianshu.com/p/34b95c5eedb6) 多人协作开发，Git Flow协同开发模式少不了。
- [鸟哥的私房菜](http://cn.linux.vbird.org/linux_basic/linux_basic.php) Linux入门必看书籍，相信我最多花一周的时间，你就能掌握90%以后常用的命令。
- [Anaconda入门使用指南](https://www.jianshu.com/p/169403f7e40c)
- [Python 2.7.x 和 3.x 版本的重要区别](https://www.techug.com/post/the-difference-of-python2-and-python3.html)
- [阿里巴巴Java开发手册终极版v1.3.0](https://files-cdn.cnblogs.com/files/han-1034683568/%E9%98%BF%E9%87%8C%E5%B7%B4%E5%B7%B4Java%E5%BC%80%E5%8F%91%E6%89%8B%E5%86%8C%E7%BB%88%E6%9E%81%E7%89%88v1.3.0.pdf) 重点关注一下编程规约部分。
- [C心得](program/C心得.pdf) 这是本科搞算法比赛那段时间总结的，用C撸代码，记录一些心得和常用的函数等。

## 论文写作


- [沈向洋：读论文的三个层次](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247501483&idx=1&sn=9b21f8e62fa2b4b33045900a1e721d30&chksm=9094cf38a7e3462ed5901bd8b0b8ebf99a892b31d75aa6eaf8b3c8cc7698b1d708fd0891ab3e&scene=0&xtrack=1&key=4154906fe631b34d38dd84006b7684c26b81516164b0db1ad41ed88e6b295a6b2c51f84b6e722a0d32a86cea9c417ee68b68cd81ed81f54d5f9b0bd8a312e6a0af10c3c39d4e36fe9c6dbc0065a88b6f&ascene=14&uin=MjU4MjA2NDQxMA%3D%3D&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=AYYm6Tp2Vj3c7ltB6nmt64I%3D&pass_ticket=q5HN3l%2Fjp0A9ZEhbo9%2BJ7Q8rb928KQ0pmv2cVGsT1W%2BtzONBEKHpJxh8UaoaMSi9)  1、快速阅读：划分结构层次；2、仔细阅读：批判思维；3、创造性阅读：积极思考。
- [如何阅读学术文献](paperwriting/如何阅读学术文献.pdf)
- [以AM为例谈谈两种研究创新模式](https://blog.csdn.net/malefactor/article/details/50583474) 分别是：应用创新（已有模型引入到新任务）；模型创新（对模型本身的改进）。
- [清华刘知远-如何写一篇合格的NLP论文](https://zhuanlan.zhihu.com/p/58752815) 
- [宗老师教你写毕业论文](http://www.nlpr.ia.ac.cn/cip/ZongReportandLecture/Reports/2014.02.27%20Writing.pdf) 中科院自动化所 宗成庆老师手把手教你写毕业论文。
- [中国科学院兰艳艳之《论文写作小白的成长之路》](https://zhuanlan.zhihu.com/p/135989892) B站地址：[戳这里](https://www.bilibili.com/video/BV1Up4y1C7cK)
- [施一公：如何提高英文的科研写作能力](http://www.cas.cn/xw/zjsd/201008/t20100812_2923299.shtml)
- [制作清晰简洁的学术汇报PPT只需学会这几招](paperwriting/制作清晰简洁的学术汇报PPT只需学会这几招.pdf)  列举了两个典型的PPT，说明学术汇报PPT的制作的要点。
- [如何审稿](paperwriting/如何审稿.pdf) 从审稿人的角度反思自己论文写作。
## 其他

- [提问的智慧](https://www.dianbo.org/9238/stone/tiwendezhihui.htm) 好的提问方式会让别人更乐意帮助你，你的问题就会更快得到解答。
- [知其所以然之永不遗忘的算法](https://www.jianshu.com/p/4a85875bf9cb)  我们以为自己已经掌握了某些算法，其实只不过是在背诵别人发明的算法，就像我们背诵历史书上的那些事件一样，时间久了就会遗忘。**只有探寻算法背后那“曲径通幽”的思维之路，并且经历了思维之路的磨难，才说得上你真正懂了算法。**
- [我的工具箱](other/我的工具箱.pdf) 我常用软件集合
- [Http状态码手册](https://www.tutorialspoint.com/http/pdf/http_status_codes.pdf)



## 最后

感谢各位看到这里，如果觉得本文有帮助到你的话，右上角点个star吧，谢谢啦。