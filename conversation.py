from ollamaChat import creatOllamaRequest, OllamaRequestOptions, OllamaMessages
from agent import *
import random

class Invition:
    def __init__(self, agent:Agent, otheragent:Agent, time: float) -> None:
        self.agent = agent
        self.otheragent = otheragent
        self.time = time
    
    def tryInvition(self) -> bool:
        """
        agent尝试发出邀请otheragent参与conversation中\n
        返回一个bool值表示是否成功相应
        """
        ret = True
        if self.agent.lastInviteAttempt == self.otheragent.name:
            # 假如其为上一个谈话对象,则拒绝?
            # 或者来减小对应的可能性会好一点?
            # 或者不愿意接受相应的对话,对每一个用户而言.
            ret = False
        if random.random() < self.agent.invitePosiblity:
            ret = False
        return ret
    


class Conversation:

    agent : Agent # 对话发起人
    otheragent : Agent # 对话接收人
    messages : list[dict] # 含有一个字典
    time : float
    maxConversationNum = 5
    maxConversationTime = 20.0 # 先随便设置一下

    def __init__(self, agent:Agent, otheragent:Agent, time: float) -> None:
        self.agent = agent
        self.otheragent = otheragent
        self.messages = []
        self.time = time
    
    def setMaxConversationNum(self, num: int) -> None:
        """
        设置最大对话数量
        """
        self.maxConversationNum = num

    def setMaxConversationTime(self, time: float) -> None:
        """
        设置最长对话时间
        """
        self.maxConversationTime = time

    def beginConversation(self) -> None:
        """
        二者开始进行对话,其结果保存在self.messages中.
        """
        self.startConversationMessage(turn=False)
        turn = True
        while not self.isStopConversation():
            self.continueConversationMessage(turn=turn)
            turn = not turn
        self.leaveConversationMessage(turn)


    def isStopConversation(self) -> bool:
        """
        会返回一个bool值
        表示当前状态下是否应该leaveConversation
        后续这个也许需要进行重构
        因为ai-town的写法非常奇怪
        """
        return len(self.messages) > self.maxConversationNum
        

    def startConversationMessage(self, turn:bool = False) -> None:

        if turn:
            agent, otheragent = self.otheragent, self.agent
        else:
            agent, otheragent = self.agent, self.otheragent

        prompt = f'You are {agent.name}, and you just started a conversation with {otheragent.name}.\n'
        prompt += self.agentPrompts(turn) + self.relatedMemoriesPrompt(turn) + agent.name + ':'
        print(prompt)
        options = OllamaRequestOptions()
        options.addstop([f'{agent.name}:',f'{otheragent.name}:'])
        messages = OllamaMessages(prompt)

        self.addMessage(creatOllamaRequest(messages, options), self.time, turn)


    def continueConversationMessage(self, turn:bool = False) -> None:

        if turn:
            agent, otheragent = self.otheragent, self.agent
        else:
            agent, otheragent = self.agent, self.otheragent

        prompt = f'You are {agent.name}, and you are currently in a conversation with {otheragent.name}.\n'
        prompt += self.agentPrompts(turn) + self.relatedMemoriesPrompt(turn) 
        prompt += f'Below is the current chat history between you and {otheragent.name}.\n DO NOT greet them again. DO NOT repeat your said before. Do NOT use the word "Hey" too often. Your response should be brief and within 200 characters.\n'
        prompt += self.conversationToString()

        print(prompt)
        options = OllamaRequestOptions()
        options.addstop([agent.name+':', otheragent.name+':'])

        messages = OllamaMessages(prompt)
        messages.append(agent.name + ':')

        self.addMessage(creatOllamaRequest(messages, options), self.time, turn)

    def leaveConversationMessage(self, turn:bool = False) -> None:
        if turn:
            agent, otheragent = self.otheragent, self.agent
        else:
            agent, otheragent = self.agent, self.otheragent
        
        prompt = f"You are {agent.name}, and you're currently in a conversation with {otheragent.name}.\n"
        prompt += "You've decided to leave the question and would like to politely tell them you're leaving the conversation.\n"
        prompt += self.agentPrompts(turn)
        prompt += f"Below is the current chat history between you and {otheragent.name}.\n"
        prompt += f"How would you like to tell them that you're leaving? Your response should be brief and within 200 characters.\n"
        prompt += self.conversationToString()

        options = OllamaRequestOptions()
        options.addstop([agent.name+':', otheragent.name+':'])

        messages = OllamaMessages(prompt)
        messages.append(agent.name + ':')

        self.addMessage(creatOllamaRequest(messages, options), self.time, turn)

        self.agent.rememberConversation(self.conversationToDict())
        self.otheragent.rememberConversation(self.conversationToDict(turn=True))

    def agentPrompts(self, turn:bool=False) -> str:
        """
        获得关于对话生成的prompt.
        """
        if turn:
            agent, otheragent = self.otheragent, self.agent
        else:
            agent, otheragent = self.agent, self.otheragent

        prompt = f'About you: {agent.description} \n '
        prompt += f'Your goals for the conversation: {agent.plan}'
        prompt += f'About {otheragent.name}: \n {otheragent.description}'
        prompt += '\n'
        return prompt
    
    def relatedMemoriesPrompt(self, turn:bool = False) -> str:
        if turn:
            agent, otheragent = self.otheragent, self.agent
        else:
            agent, otheragent = self.agent, self.otheragent

        prompt = 'Memories: \n'
        
        prompt += '\n'.join(agent.getMemoryAbout(otheragent.name, 3))
        # 读取otheragent相同的记忆
        # 以agent的视角来回忆
        return prompt

    def addMessage(self, message:str, time:float, turn:bool = False, ) -> None:
        """
        添加message,
        turn = true 表明两者身份对调
        为otheragent对agent的回答
        反之则为 agent 对 otheragent
        """
        if turn:
            agent, otheragent = self.otheragent, self.agent
        else:
            agent, otheragent = self.agent, self.otheragent

        message = {
            'time':time,
            'agent':agent.name,
            'otheragent':otheragent.name,
            'message':message,
            }
        self.messages.append(message)
        print(message['agent'] + ':' + message['message'] + '\n')

    
    def conversationToString(self) -> str:
        """
        将conversation的内容转换为str用于prompt输入
        """
        ret = ''
        for message in self.messages:
            ret += message['agent'] + ':' + message['message'] + '\n'
        ret += '\n'
        return ret

    def conversationToList(self) -> list[dict]:
        """
        将conversation以list[dict]的形式来返回
        """
        return self.messages
    
    def conversationToDict(self, turn:bool=False) -> dict:
        if turn:
            agent, otheragent = self.otheragent, self.agent
        else:
            agent, otheragent = self.agent, self.otheragent

        conversation = {
            'agent':agent.name,
            'time':self.time,
            'otheragent':otheragent.name,
            'content':self.conversationToString(),
            'embedding':None,
        }
        return conversation


if __name__ == '__main__':
    description = """
    Kira wants everyone to think she is happy. But deep down,
    she's incredibly depressed. She hides her sadness by talking about travel,
    food, and yoga. But often she can't keep her sadness in and will start crying.
    Often it seems like she is close to having a mental breakdown.    
    """
    plan = """
    You want find a way to be happy.
    """
    agent1 = Agent(name='Kira', description=description, plan=plan, location=Agentloaction(0,0))
    description = """
    Pete is deeply religious and sees the hand of god or of the work
    of the devil everywhere. He can't have a conversation without bringing up his
    deep faith. Or warning others about the perils of hell.
    """
    plan = 'You want to convert everyone to your religion.'
    agent2 = Agent(name='Pete', description=description, plan=plan, location=Agentloaction(0,0))

    createmessage = Conversation(agent1, agent2, time=0)
    createmessage.beginConversation()
    # agent1.review()
    agent1.saveData('./Localizaltion/data/')
    agent2.saveData('./Localizaltion/data/')
    agent1.review()