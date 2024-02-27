from agent import Agent
from conversation import *
from embedder import embedder
from data import Load,Save
from options import Options
import os


class World:

    time : float            # 世界现在处于的时间
    agents : list[Agent]    # Agent群
    name : str              # 世界名称
    Config : Options

    def __init__(self, name:str='World') -> None:
        """
        初始化世界参数,时间归零 \n
        分档归类World
        """
        self.agents = Load.loadAgent()
        self.time = 0
        self.name = name
        self.embedder = embedder()


    def loadworld(self) -> None:
        """
        从文件中加载world
        """
        pass

    def _step(self) -> None:
        """
        世界向前step一次时间周期
        """
        self.time += 1
        self._conversation()
        self._remember()
        self._review()

    def run(self, time) -> None:
        """
        世界向前运行 time 次
        """
        for i in range(time):
            self._step()

    def cleardata(self) -> None:
        """
        清除所有数据
        """
        pass

    def _conversation(self) -> None:
        """
        生成conversation
        """
        for agent in self.agents:
            pass

    def _remember(self) -> None:
        """
        agent记忆conversation
        """
        pass

    def _review(self) -> None:
        """
        agent进行review来产生insight
        """
        pass

    def save(self) -> None:
        """
        保存世界文件
        """
        pass

if __name__ == "__main__":
    world = World()