from ollamaChat import Prompt, createOllamaRequest, OllamaRequestOptions, OllamaMessages
from agent import Agent, Agentloaction
import numpy as np
import random
import time
import pandas as pd
import os

class Invition:
    def __init__(self, agent:Agent, otheragent:Agent) -> None:
        self.agent = agent
        self.otheragent = otheragent
        self.time = time.time()
    
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


class ConversationMessage:
    def __init__(self, sender:Agent, recipient:Agent, chatHistory:str,
                options:OllamaRequestOptions) -> None:
        self.sender = sender
        self.recipient = recipient
        self.message = None
        self.chatHistory = chatHistory
        self.prompts = []
        self.options = options
        self.stopConversation:bool

    def createMessage(self) -> dict:
        ollamaMessages = OllamaMessages(self.prompts)
        message = createOllamaRequest(ollamaMessages, self.options)
        self.message = message
        return self.toDict()
    
    def relatedPrompt(self) -> str:
        ret = Prompt([
            self.agentPrompt(),
            self.memoryPrompt(),
        ]).join()
        return ret

    def agentPrompt(self) -> str:
        ret = Prompt([
            f'Your description: {self.sender.description}',
            f'Your plan in this conversation: {self.sender.plan}',
            f'About your recipient: {self.recipient.description}',
        ]).join()
        return ret

    def memoryPrompt(self) -> str:
        if self.chatHistory != '':
            memory = self.sender.getMemoryAbout(3, self.chatHistory)
            if len(memory) != 0:
                ret = Prompt([
                    f'Your memory about {self.recipient.name}:',
                    Prompt(memory).join(),
                ]).join()
            else:
                ret = ''
        else:
            ret = ''
        return ret
        
    def __str__(self) -> str:
        if self.message == None:
            print("message is not created")
        return self.sender.name + ' to '+ self.recipient.name + ': ' + self.message
    
    def toDict(self) -> dict:
        ret = {
            'sender': self.sender.name,
            'recipient': self.recipient.name,
            'message': self.message,
            'time': time.time(),
        }
        return ret

    def isStopConversation(self) -> bool:
        prompt = Prompt([
            f"You are {self.sender.name}, now you are chatting with {self.recipient}.",
            self.relatedPrompt(),
            "Do you want to end this conversation?",
            "Your answer must be 'Y' or 'N', And only have one character. e.g: \"Y\"",
            f"Below is the conversation history with {self.recipient.name}:",
        ]).join()
        ollamaMessages = OllamaMessages([prompt, self.chatHistory,'You:'])
        options = OllamaRequestOptions().setOptions(num_predict=1,temperature=0.7,top_k=20,top_p=0.7)
        ret = createOllamaRequest(ollamaMessages,options)
        return ret != 'Y'
    
    def update(self, chatHistory:str):
        self.chatHistory = chatHistory
        self.sender, self.recipient = self.recipient, self.sender
        return self




# 三种ConversationMessages有着相似的定义,类似的方法

class StartConversationMessage(ConversationMessage):
    def __init__(self, sender:Agent, recipient:Agent, chatHistory:str,
                options:OllamaRequestOptions) -> None:
        super().__init__(sender, recipient, chatHistory, options)
        prompt = Prompt([
            f'You are {sender.name}, and you just started a conversation with {recipient.name}.',
            self.agentPrompt()
        ])

        self.prompts = [prompt.join(), self.sender.name+':']


class ContinueConversationMessage(ConversationMessage):
    def __init__(self, sender: Agent, recipient: Agent, chatHistory:str,
                 options: OllamaRequestOptions) -> None:
        super().__init__(sender, recipient, chatHistory, options)
        prompt = Prompt([
            f'You are {sender.name}, and you are currently in a conversation with {recipient.name}.',
            self.relatedPrompt(),
            f'Below is the current chat history between you and {recipient.name}.\n DO NOT greet them again. DO NOT repeat your said before. Do NOT use the word "Hey" too often. Your response should be brief and within 200 characters.',            
        ])
        self.prompts = [
            prompt.join(), 
            self.chatHistory,
            self.sender.name+':'
        ]
    
    

class EndConversationMessage(ConversationMessage):
    def __init__(self, sender:Agent, recipient:Agent, chatHistory:str, 
                 options:OllamaRequestOptions) -> None:
        super().__init__(sender, recipient, chatHistory, options)
        prompt = Prompt([
            f"You are {sender.name}, and you're currently in a conversation with {recipient.name}.",
            "You've decided to leave the question and would like to politely tell them you're leaving the conversation.",
            self.agentPrompt(),
            f"Below is the current chat history between you and {recipient.name}.",
            f"How would you like to tell them that you're leaving? Your response should be brief and within 200 characters.",
        ])
        self.prompts = [
            prompt.join(),
            chatHistory,
            sender.name+':',
        ]

# 集成一个总类型

class Conversation:

    agent : Agent # 对话发起人
    otheragent : Agent # 对话另一个
    messages : list[dict] # 含有一个字典
    maxConversationNum = 8
    maxConversationTime = 2000.0 # 先随便设置一下
    ollamaRequestOptions = OllamaRequestOptions().setOptions(temperature=0.9,repeat_penalty=1.5,top_p=0.95,top_k=100)

    def __init__(self, agent:Agent, otheragent:Agent) -> None:
        self.agent = agent
        self.otheragent = otheragent
        self.messages = []
        self.time = time.time()
        self.ollamaRequestOptions.setStopWord([f'{agent.name}:',f'{otheragent.name}:'])

    def stepConversation(self) -> None:
        """
        二者开始进行对话,其结果保存在self.messages中.
        """

        endConversation = False

        startConversationMessage = StartConversationMessage(
            self.agent,
            self.otheragent,
            self.toFormattedString(),
            self.ollamaRequestOptions,
            )
        self.pushMessage(startConversationMessage.createMessage())

        continueConversationMessage = ContinueConversationMessage(
            self.otheragent,
            self.agent,
            self.toFormattedString(),
            self.ollamaRequestOptions,
        )

        self.pushMessage(continueConversationMessage.createMessage())
        
        def isEndConversation() -> bool:
            j1 = len(self.messages) > self.maxConversationNum
            j2 = (time.time() - self.time) > self.maxConversationTime
            return j1 or j2

        while not endConversation :
            continueConversationMessage.update(self.toFormattedString())
            self.pushMessage(continueConversationMessage.createMessage())
            endConversation = isEndConversation() or continueConversationMessage.isStopConversation()

        endConversationMessage = EndConversationMessage(
            continueConversationMessage.recipient,
            continueConversationMessage.sender,
            self.toFormattedString(),
            self.ollamaRequestOptions,
        )
        self.pushMessage(endConversationMessage.createMessage())
        endConversationMessage.update(self.toFormattedString())
        self.pushMessage(endConversationMessage.createMessage())

        conversationForAgent, conversationForOtherAgent = self.toDict()

        self.agent.rememberConversation(conversationForAgent)
        self.otheragent.rememberConversation(conversationForOtherAgent)            

    def pushMessage(self, message:dict) -> None:
        self.messages.append(message)
    
    def toFormattedString(self) -> str:
        if len(self.messages) != 0:
            ret = []
            for message in self.messages:
                ret.append(f"{message['sender']} to {message['recipient']}: {message['message']}")
            ret = '\n'.join(ret)
        else: 
            ret = ''
        return ret
    
    def toDict(self):
        conversationForAgent = {
            'agent':self.agent.name,
            'otheragent':self.otheragent.name,
            'content':self.messages,
        }
        conversationForOtherAgent = {
            'agent':self.otheragent.name,
            'otheragent':self.agent.name,
            'content':self.messages,
        }
        return conversationForAgent, conversationForOtherAgent

    def save(self) -> None:
        path = '/home/tenghao/Localizaltion/data/'
        existedData = pd.read_csv(path + "messages.csv",index_col=0)
        df = pd.DataFrame(self.messages)
        pd.concat(existedData, df, ignore_index=True)
        existedData.to_csv(path + "messages.csv")

    def __del__(self) -> None:
        self.save()


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