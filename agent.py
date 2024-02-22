import pandas as pd
from ollamaChat import creatOllamaRequest, OllamaRequestOptions, OllamaMessages
import embedder


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
    conversations : pd.DataFrame
    location : Agentloaction
    lastInviteAttempt : str
    inProgressOperation: dict[str:str, str:str, str:float] # 表明现在正在执行的操作
    """
    {
        name: string;
        operation: string;
        started: number;
    }
    """

    def __init__(self, name:str, description:str, plan:str, location:Agentloaction) -> None:
        # 对Agent的内容进行初始化
        self.name = name
        self.description = description
        self.plan = plan
        self.memories = pd.DataFrame(columns=['agent','time','otheragent','content','embedding','type','importance'])
        self.allmemories = pd.DataFrame(columns=['agent','time','otheragent','content','embedding','type','importance'])
        self.conversations = pd.DataFrame(columns=['agent','time','otheragent','content','embedding'])
        self.location = location
       
    def rememberConversation(self, conversation:dict) -> None:
        # 将conversation加入到conversation数据库中
        self.conversations.loc[len(self.conversations)] = conversation.values
        
        # 将conversation进行summary后进行remember操作
        prompt = f"""You are {conversation['agent']}, and you just finished a conversation with {conversation['otheragent']}. I would
            like you to summarize the conversation from {conversation['agent']}'s perspective, using first-person pronouns like
            "I," and add if you liked or disliked this interaction. \n"""
        prompt += conversation['content']
        options = OllamaRequestOptions()
        messages = OllamaMessages(prompt)

        messages.append('Summary:')
        memory = creatOllamaRequest(messages, options)
        embedding = embedder.embeddingForOne(memory)
        importance = self.calculateImportance(memory)

        # 'agent','time','otherAgent','content','embedding','type','importance'
        self.memories.loc[len(self.memories)] = [conversation['agent'],
                                                 conversation['time'],
                                                 conversation['otheragent'],
                                                 memory,
                                                 embedding,
                                                 'memory',
                                                 importance,]
        
        self.allmemories.loc[len(self.memories)] = [conversation['agent'],
                                                 conversation['time'],
                                                 conversation['otheragent'],
                                                 memory,
                                                 embedding,
                                                 'memory',
                                                 importance,]
    
    def sortMemory(self) -> None:
        """
        按照importance对memory进行排序
        """
        self.memories.sort_values(by='importance', ascending=False, inplace=True)

    def getMemory(self, num:int) -> list[str]:
        """
        从memory中获得最近几条比较重要的memory
        """
        self.sortMemory()
        ret = self.memories['content'].head(num).to_list()
        return ret
    
    def forgetMemory(self, max:int) -> None:
        """
        超过max num的记忆会被遗忘,所有记忆存储在allmemories里面
        """
        self.sortMemory()
        self.memories = self.memories.head(max)


    def getMemoryAbout(self, name:str, num:int) -> list[str]:
        """
        获得关于某人的记忆:
        name: otheragent的姓名
        """
        # 个人认为About不如是直接增加importance的值,然后从最大中选择出前max个
        self.sortMemory()
        memories = self.memories[self.memories['otheragent'] == name]
        ret = memories['content'].head(num).to_list()
        return ret

    def calculateImportance(self, memory:str) -> str:
        prompt = "On the scale of 0 to 9, where 0 is purely mundane (e.g., brushing teeth, making bed) and 9 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following piece of memory.\n"
        prompt += 'Memory:' + memory + '\n'
        prompt += "Answer on a scale of 0 to 9. Respond with number only, e.g. \"5\""

        options = OllamaRequestOptions()
        options.setMaxToken(1)

        messages = OllamaMessages(prompt)

        # 目前我无法确保这里面得到的值一定是0~9
        importance = creatOllamaRequest(messages, options)
        return importance
    
    def review(self) -> None:
        prompt = ['[no prose]', '[Output only JSON]', f'You are {self.name}, statements about you:']
        memories = self.memories['content'].to_list()
        for memory in memories:
            prompt.append('Statement:' + memory)
        prompt.append("What 3 high-level insights can you infer from the above statements?")
        prompt.append("""Return in JSON format, where the key is a list of input statements that contributed to your insights and value is your insight. Make the response parseable by Typescript JSON.parse() function. DO NOT escape characters or include "\n" or white space in response.""")

    def save(self) -> None:
        self.conversations.to_csv(f'data/[{self.name}][conversation].csv')
        self.allmemories.to_csv(f'data/[{self.name}][memories].csv')









        



    

