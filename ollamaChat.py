import ollama

# python 为 ollama 提供了一个非常便捷的库
# 看了一下应该只是将ollama进行了打包操作,放置在python内进行使用的一个库
# 其中似乎并不能设置ollama的baseurl

# messages:list, options, model='llama2'

class OllamaRequestOptions:
	"""
	创建一个options便于之后使用
	默认为: stop:['user:']
	"""
	# https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
	def __init__(self) -> None:
		self.options = {
		# "num_keep": 2,
		# "num_predict": 100, # 限制预测量, 我也不清楚这个应该怎么设置了 
		# "top_k": 40, # 生成更多样化的答案?
		# "top_p": 0.9,
		# "tfs_z":1,
		# "repeat_last_n": 64,
		# "temperature": 0.8,
		# "repeat_penalty": 1.1, # 重复惩罚 
		# "mirostat": 0,
		# "mirostat_tau": 5.0,
		# "mirostat_eta": 0.1,
		"stop": ["user:"],
		"num_ctx": 2048, # 默认记忆大小
		}

	def toDict(self) -> dict:
		"""
		获得一个默认的options类
		"""
		return self.options
	
	def setStopWord(self, stopWord:list[str]) -> None:
		"""
		添加一组stop参数到options中
		"""
		for word in stopWord:
			self.options['stop'].append(word)

	def setOptions(self, top_k=40, top_p=0.9, tfs_z=1, repeat_last_n=64,
				temperature=0.8,repeat_penalty=1.1,mirostat=0,mirostat_tau=5.0,
				mirostat_eta = 0.1,num_predict=128, ):
		self.options['top_k'] = top_k
		self.options['top_p'] = top_p
		self.options['tfs_z'] = tfs_z
		self.options['repeat_last_n'] = repeat_last_n
		self.options['temperature'] = temperature
		self.options['repeat_penalty'] = repeat_penalty
		self.options['mirostat'] = mirostat
		self.options['mirostat_tau'] = mirostat_tau
		self.options['mirostat_eta'] = mirostat_eta
		self.options['num_predict'] = num_predict
		return self

class OllamaMessages:
	"""
	创建一个messages类便于之后使用
	"""
	def __init__(self, prompts:list[str]) -> None:
		self.messages = []
		for prompt in prompts:
			self.messages.append({
				'role':'user',
				'content': prompt,
			})
	
	def toList(self) -> list:
		return self.messages
	
	
class Prompt:
	
	prompt:list[str]

	def __init__(self, content:list[str]=[]) -> None:
		self.prompt = content

	def append(self, content:list[str]) -> None:
		self.prompt += content

	def join(self, char:str='\n') -> str:
		return char.join(self.prompt)
	
	def __add__(self, prompt):
		self.prompt.append(prompt)
		return self

	def __iadd__(self, prompt):
		self.prompt.append(prompt)
		return self
	def __str__(self) -> str:
		return self.join()
	

def createOllamaRequest(messages:OllamaMessages, options:OllamaRequestOptions, model='llama2') :

	"""
	向ollama请求返回
	messages : 为一个列表,含有 'role','content'等属性
	options : 包含对于该请求的设置,如stop, maxtoken
	model : 默认使用llama2:7b来进行生成
	"""
	response = ollama.chat(model, messages.toList(), stream=False, options=options.toDict())
	ret = response['message']['content']
	print(ret)
	return ret

if __name__ == '__main__':

	message = OllamaMessages(["why is the sky blue?"])
	option = OllamaRequestOptions()
	ret = createOllamaRequest(message, option)
	print(ret)

"""
Ollama的返回格式
因此只需要其中message钟content部分的内容?
{	'model': 'llama2',
 	'created_at': '2024-02-14T10:16:43.183253778Z',
	'message': {
		'role': 'assistant', 
		'content': "\nThe sky appears blue because of a phenomenon called Rayleigh scattering, which occurs when light travels through the Earth's atmosphere. When sunlight enters the atmosphere, it encounters tiny molecules of gases such as nitrogen and oxygen. These molecules scatter the light in all directions, but they scatter shorter (blue) wavelengths more than longer (red) wavelengths.\n\nAs a result of this scattering, the blue light is dispersed throughout the atmosphere, giving the sky its blue appearance. The red light, on the other hand, passes through the atmosphere mostly unscattered, which is why we can see the sun as a bright red ball during sunrise and sunset.\n\nThe reason for this color difference is due to the wavelength of the light. Blue light has shorter wavelengths than red light, so it is more easily scattered by the small molecules in the atmosphere. This is known as Rayleigh scattering, named after the British physicist Lord Rayleigh, who first described the phenomenon in the late 19th century.\n\nIn addition to Rayleigh scattering, there are other factors that can affect the color of the sky, such as pollution, dust, and water vapor. However, the blue color of the sky is primarily due to the scattering of sunlight by the atmosphere."},
	'done': True, 
	'total_duration': 8539149956, 
	'load_duration': 2961477971, 
	'prompt_eval_count': 26, 
	'prompt_eval_duration': 66420000, 
	'eval_count': 296, 
	'eval_duration': 5505678000}
"""