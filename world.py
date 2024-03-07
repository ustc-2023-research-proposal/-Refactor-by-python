from agent import Agent
from conversation import Conversation
from embedder import Embedder
from data import Load,Save,Config
import time
import numpy as np
import os


class World:

    agents : list[Agent]    # Agent群
    name : str              # 世界名称
    dataPath = Config.datapath
    embedder : Embedder

    def __init__(self, worldName:str='World') -> None:
        """
        初始化世界参数,时间归零 \n
        分档归类World
        """
        self.agents = Load.loadAgent(self.dataPath)
        self.worldName = worldName
        self.embedder = Embedder()
        self.createdTime = time.time()

    def _step(self) -> None:
        """
        世界向前step一次
        """
        self._conversation()

    def run(self, times) -> None:
        """
        世界向前运行 times 次
        """
        for i in range(times):
            self._step()

    def _conversation(self) -> None:
        """
        生成conversation
        """
        conversation = Conversation(self.agents[0], self.agents[1])
        conversation.stepConversation()
        self.agents[0].saveData(self.dataPath)

    def _review(self) -> None:
        """
        agent进行review来产生insight
        """
        pass

    def save(self) -> None:
        Save.saveWorld(self)
        

class Application():

    endApplication = False
    world: World
    dataPath = Config.datapath

    def __init__(self) -> None:
        inst = {
            "i":self.init,
            "q":self.quit,
            "s":self.save,
            "l":self.load,
            "r":self.run,
            "list":self.printList,
            }
        prompt = ""
        print(prompt)

        while self.endApplication:
            command = input("Please input the command you want to execute.")
            try:
                inst[command]()
            except KeyError:
                print("Command not found. Please input again")

    def init(self) -> None:
        worldName = input("Please the input the name you want to create.")
        self.world = World(worldName)

    def quit(self) -> None:
        self.endApplication = True
    
    def load(self) -> None:
        self.world = World(input("Please the name you want to load"))
    
    def save(self) -> None:
        self.world.save()

    def run(self) -> None:
        self.world.run(1)
    
    def printList(self) -> None:
        dir = os.listdir(self.dataPath)
        print("ExistWorld:")
        for i in range(len(dir)):
            print(i,'. ',dir[i])


if __name__ == "__main__":
    app = Application()
            

