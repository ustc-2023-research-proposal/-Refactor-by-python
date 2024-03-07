import pandas as pd
from agent import *
import json

class Config:
    # Data Structure
    """
    - data
        - world
            - conversation
            - memory
        - agent  
    """
    datapath = '/home/tenghao/Localizaltion/data/'


    maxConversationNum : int = 5
    maxConversationTime : float = 2000

    inferanceModel : str = 'llama2:7b'
    invitePossibility : float = 0.8

    maxMemoryNum : int = 10

    def __init__(self) -> None:
        pass

    def setDataPath(self, path:str) -> None:
        self.datapath = path
     

class Load:

    def loadAgent() -> list[Agent]:
        """
        从datapath中读取初始化数据\n
        以用于生成所有agent
        返回一个拥有agent的list
        """
        path = Config.datapath + 'Agents.csv'
        df = pd.read_csv(path, index_col=0)
        agents = []
        for i in range(len(df.index)):
            location = Agentloaction(x=0, y=0)
            agent = Agent(df.iloc[i]['name'], df.iloc[i]['description'], df.iloc[i]['plan'],location)
            agents.append(agent)
        print(f"Successful to load and Init {len(df.index)} Agents.")
        return agents

    def loadAgentData(agents:list[Agent]) -> None:
        """
        从data/csv中读取所有conversations以及memories
        agents: 一个list,包含所有agent
        """
        for agent in agents:
            agent.loadData(path=Config.datapath)

    def loadWorld() -> None:
        return pd.read_pickle(Config.datapath)

    def loadConfig(self, path:str) -> Config:
        config = pd.read_json(path)
        return config


class Save:
    datapath = "data/"

    def saveAgentData(self, agents:list[Agent]) -> None:
        """
        将AgentData保存到指定目录下
        """
        for agent in agents:
            agent.saveData(path=self.datapath)

    def saveWorld(self, world) -> None:
        """
        将世界保存为plk
        """
        pd.to_pickle(world, self.datapath + f"{world.worldName}.pkl")

    def saveConfig(self, config:Config, path:str) -> None:
        pass