import pandas as pd
from ollamaChat import Prompt, createOllamaRequest, OllamaRequestOptions, OllamaMessages
from embedder import Embedder
import numpy as np
import ast
import time # time.asctime() 返回一个当前时间戳


class Agentloaction:
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

    def getLocation(self) -> tuple[float,float]:
        return (self.x, self.y)
    
    def setLocation(self,x,y) -> tuple[float,float]:
        self.x = x
        self.y = y
        return (self.x, self.y)


class Agent:
    name : str
    description : str
    plan : str
    memories: pd.DataFrame
    conversations : list[dict] # 设置最大限额量,并且一旦超过量就写入文件中
    location : Agentloaction
    lastInviteAttempt : str # 上一次尝试邀请的agent的姓名
    invitePosiblity : float # 表明其接受invite的可能性大小
    inProgressOperation: dict[str:str, str:str, str:float] # 表明现在正在执行的操作
    maxMemoryNum : int
    

    """
    {
        name: string;
        operation: string;
        started: float;
    }
    """

    def __init__(self, name:str, description:str, plan:str, location:Agentloaction) -> None:
        # 对Agent的内容进行初始化
        self.name = name
        self.description = description
        self.plan = plan
        self.memories = pd.DataFrame(columns=['agent','time','content','type','importance','embedding'])
        self.conversations = []
        self.location = location
        self.invitePosiblity = 0.8 # 这个可能是需要随着进程变动的

       
    def rememberConversation(self, conversation:dict) -> None:
        # 将conversation加入到conversation数据库中
        
        def getConversationContent(conversation:dict) -> str:
            ret = []
            massages = conversation['content']
            for massage in massages:
                ret.append(massage['sender']+' to '+massage['recipient']+': '+massage['message'])
            return '\n'.join(ret)
        
        # 将conversation进行summary后进行remember操作
        prompt = Prompt([
        f"""You are {conversation['agent']}, and you just finished a conversation with {conversation['otheragent']}. I would
            like you to summarize the conversation from {conversation['agent']}'s perspective, using first-person pronouns like
            "I," and add if you liked or disliked this interaction.""",
            getConversationContent(conversation) 
        ])
    
        options = OllamaRequestOptions()
        messages = OllamaMessages([prompt.join(), self.name+':'])

        memory = createOllamaRequest(messages, options)
        embedding = Embedder.embeddingForOne(Embedder.model, memory)
        importance = self.calculateImportance(memory)

        self.memories.loc[len(self.memories)] = [
            conversation['agent'],
            time.time(),
            memory,
            'memory',
            importance,
            embedding,
        ]


    def sortMemory(self, conversationEmbedding) -> None:
        """
        按照importance对memory进行降序排序
        """
        # 需要继续修正 / 对于时间项
        revelence = self.memories['time'].to_list()
        revelence = [np.e**(i-time.time()) for i in revelence]

        importance = self.memories['importance'].to_list()
        importance = [int(i) for i in importance]

        memoryEmbedding = self.memories['embedding'].to_list()
        memoryEmbedding = [ast.literal_eval(i) for i in memoryEmbedding]
        
        similarity = Embedder.caculateSimilarity(memoryEmbedding,conversationEmbedding)
        
        score = [i+j+z for i,j,z in zip(importance,similarity,revelence)]
        self.memories['score'] = score

        self.memories.sort_values(by='score', ascending=False, inplace=True)

        
    
    def forgetMemory(self, max:int) -> None:
        """
        超过max num的记忆会被遗忘,所有记忆存储在allmemories里面
        """
        self.memories.sort_values(by='score', ascending=False, inplace=True)
        self.memories = self.memories.head(max)

    def getMemoryAbout(self, num:int, chatHistory:str) -> list[str]:
        """
        num: num of requiring
        chatHistory: str instead of list[str].
        """
        if len(self.memories.index) == 0:
            ret = []
        else: 
            conversationEmbedding = Embedder.embeddingForOne(Embedder.model,chatHistory)
            self.sortMemory(conversationEmbedding)
            ret = self.memories.head(num)['content'].to_list()
        return ret

    def calculateImportance(self, memory:str) -> str:
        prompt = "On the scale of 0 to 9, where 0 is purely mundane (e.g., brushing teeth, making bed) and 9 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.\n"
        prompt += 'Memory:' + memory + '\n'
        prompt += "Answer on a scale of 0 to 9. Respond with number only, e.g. \"5\""

        options = OllamaRequestOptions()
        # 设置其最大输出数量为1
        options.setOptions(num_predict=1, temperature=0.7)
        messages = OllamaMessages(prompt)

        # 目前暂时以str的形式来进行存储
        importance = createOllamaRequest(messages, options)
        return importance


    def review(self, lastrespond:list[str]= None) -> None:
        prompt = Prompt([
            f"You are {self.name}.",
            f"Your decription: {self.description}",
            f"Your plan: {self.plan}",
            "Below is your memories, what can you infer from your memories. Please use 'I' in your respond. Your repond should be brief and in 100 charaters.",
            f"Memories: {self.memories}",
        ])

        if len(lastrespond) != 0:
            prompt += "Do not be similiy with it in your respond.You need to respond the defferent answer."
            for respond in lastrespond:
                prompt += respond

        ollamaMessages = OllamaMessages([prompt.join(),"Inference:"])
        options = OllamaRequestOptions().setOptions(temperature=0.7,top_k=40)
        respond = createOllamaRequest(ollamaMessages, options)

        embedding = Embedder.embeddingForOne(Embedder.model,respond)

        self.memories.iloc[len(self.memories)] = [
            self.name,
            time.time(),
            None,
            respond,
            embedding,
        ]


    def doSomethingElse(self) -> None:
        """
        假如所有对话邀请都被拒绝,并且在一轮conversation运行结束后无事可做\n
        此时会执行这个函数\n
        向ollama输入一个当前状态,然后返回一个agent想进行的动作.
        """
    
    def saveData(self, path:str) -> None:
        """
        将conversations, allmemories的内容保存至data目录下.\n
        确保path路径为str,并且以'/'结尾
        """
        self.conversations.to_csv(path + f'[{self.name}][conversations].csv')
        self.memories.to_csv(path + f'[{self.name}][memories].csv')
        self.allmemories.to_csv(path + f'[{self.name}][allmemories].csv')

    def loadData(self,path:str) -> None:
        """
        读取conversation,memories的数据 \n
        确保path路径为str,并且以'/'结尾
        """
        self.conversations = pd.read_csv(path + f'[{self.name}][conversations].csv', index_col=0)
        self.memories = pd.read_csv(path + f'[{self.name}][memories].csv', index_col=0)
        self.allmemories = pd.read_csv(path + f'[{self.name}][allmemories].csv', index_col=0)
    




        



    

