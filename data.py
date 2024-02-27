import pandas as pd
from agent import *
import json

class Config:
    datapath = 'data/'

    def __init__(self) -> None:
        pass

    def setDataPath(self, path:str) -> None:
        self.datapath = path
     

class Load:
    dataPath = "data/"

    def loadAgent(self) -> list[Agent]:
        """
        从datapath中读取初始化数据\n
        以用于生成所有agent
        返回一个拥有agent的list
        """
        path = self.dataPath + 'agentDescription.csv'
        df = pd.read_csv(path, index_col=0)
        agents = []
        for i in range(len(df.index)):
            location = Agentloaction(x=0, y=0)
            agent = Agent(df.iloc[i]['name'], df.iloc[i]['description'], df.iloc[i]['plan'], location)
            agents.append(agent)
        print(f"Successful to load and Init {len(df.index)} Agents.")
        return agents

    def loadAgentData(self, agents:list[Agent]) -> None:
        """
        从data/csv中读取所有conversations以及memories
        agents: 一个list,包含所有agent
        """
        for agent in agents:
            agent.loadData(path=self.dataPath)

    def loadWorld(self) -> None:
        """
        从世界文件中load世界
        """
        pass

class Save:
    datapath = "data/"

    def saveAgentData(self, agents:list[Agent]) -> None:
        """
        将AgentData保存到指定目录下
        """
        for agent in agents:
            agent.saveData(path=self.datapath)

    def saveWorld(self) -> None:
        pass

