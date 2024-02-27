class Options:
    # Data Load and Save
    datapath : str = "data/"

    # Conversation Options
    maxConversationNum : int = 5
    maxConversationTime : int = 120

    # Ollama Options
    inferaceModel : str = 'llama2:7b'

    # Agent Options
    invitePossibility : float = 0.8
    maxMemoryNum : int = 10

    def __init__(self) -> None:
        pass


        