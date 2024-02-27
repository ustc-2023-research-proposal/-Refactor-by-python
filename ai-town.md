# Ai-Town-Localization-for-python

## First. csv Stroge

1. 使用 `pandas` 中的 `dataframe` 框架来对数据进行处理
2. 使用csv来作为表格来进行存储,

具体所需要的格式为 when who what
具体形式为:

index|conversation|embedding|name|to

因此首先这个数据能够进行读写操作.

以及对于reply格式来进行操作的方法

## Second. Game and Conversation

1. 创建本程序到ollama的请求
    - 采用`python`的request库来进行操作
    - 其次是由于不是基于前后端交互的形式,因此不需要在每个时刻进行相应,因此可以暂时不用处理多线程的问题
    - 区分为conversation
        - 其prompth的格式,在github上有写到
    - review的发生
    - plan的产生
    - 总共三种
2. 将ollama的返回作为程序响应结果来返回到dataframe中进行存储
    - 在进行关机操作后需要将其转换为csv格式来进行存储
    - 其余时刻由于产生的数据量相对不大,因此我认为可以直接放置在内存里面
3. 具体的player进入交互过程不需要具体的操作来进行实现.

## third. others action

1. 在agent不进行交互的时候会进行其他操作,因此可以暂缓这个方面的设计过程.
2. 主要过程在于,应当设计一个主时间轴进程,比如循环的24小时设置,用户和角色的对话内容应当决定了这个agent所做的主要行为过程

## Forth. design

1. 设计过程应当依据对象化的形式来进行数据的处理过程,为此首先需要掌握ollama的具体返回过程
    - 在本程序中,ollama的返回过程是以流的形式来进行的,应当而言,流的形式会产生非常多的不必要信息来进行处理,因此我认为不是一个合适的选择方式,在其中不会发生断联的情况下,还是依照于整体的形式来进行传播.

2. 在python中的ollama的驱动方式.

## Date 2024/2/14
1.  [Descriptions](https://github.com/a16z-infra/ai-town/blob/86f1275eecf6832d0236e8f429af7c635a0a48ad/data/characters.ts), 为各个角色的初始描述内容.
    - 作为初始内容需要将其保存下来
2. 发起conversation.
    - 首先尝试直接向ollama发送该声明

3. [memory的prompt](https://github.com/a16z-infra/ai-town/blob/86f1275eecf6832d0236e8f429af7c635a0a48ad/convex/agent/memory.ts#L359),[memory的remember](https://github.com/a16z-infra/ai-town/blob/86f1275eecf6832d0236e8f429af7c635a0a48ad/convex/agent/memory.ts#L45)
    - 来创建类似的方法

4. 找到sendMessages时的prompt. [conversation prompt](https://github.com/a16z-infra/ai-town/blob/86f1275eecf6832d0236e8f429af7c635a0a48ad/convex/agent/conversation.ts#L15)

5. 因此可以直接开始尝试来写这个该如何实现了
    - 先从sendMessages开始

6. 我看了一下源码,在重构的时候,但是我发现在recall的时候他只会回忆关于这个人的对话内容,这个是不是不合理的,
不然不太可能出现邀请别人的想法? 暂时按照其来写?

7. ![暂时跑了一下的结果](image.png), 这是使用player的初始描述和plan来跑出来的对话结果
    - 非常简化的版本,其中不包含一些信息,如位置和时间.
    - 还没有实现recall
    - 事件embedding需要加的位置暂时不是很清楚.

8. conversation开始后是怎么判断其需要结束了的呢?
    - 看了一下, 对话时间超过两分钟或者,对话总数超过八条之后就会结束,
    - 但是这个两分钟是现实世界的两分钟?
    - 所以说在对话尚未生成结束之前这个应该如何进行操作
    - 这个结束和斯坦福那篇文章产生了一点区别
    - 我这边暂时删除了time的概念,


9. `prompt.push(...previousConversationPrompt(otherPlayer, lastConversation));`
    - 在写的时候看到这样一个写法很奇怪,这边表示他会提取出有关于之前与他对话的内容?
    - 按正常的思路来说应该时得到上次对话的记忆才对.
    - 我看了一下具体的代码,其结果只是返回一个时间参数,并且这个时间参数非常奇怪.
    - 于是我将这部分去掉了

