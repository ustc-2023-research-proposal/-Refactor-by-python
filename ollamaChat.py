import ollama
import json
# python 为 ollama 提供了一个非常便捷的库
# 看了一下应该只是将ollama进行了打包操作,放置在python内进行使用的一个库
# 其中似乎并不能设置ollama的baseurl

model = 'llama2' # 默认使用llama2来进行chatGenerate
# messages:list, options, model='llama2'


class OllamaRequestOptions:
	"""
	创建一个options便于之后使用
	默认为: stop:['user:']
	"""
	# https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
	def __init__(self) -> None:
		self.dict = {
		# "num_keep": 2,
		# "seed": 42,
		# "num_predict": 100, # 限制预测量, 我也不清楚这个应该怎么设置了 
		# "top_k": 20,
		# "top_p": 0.9,
		# "tfs_z": 0.5,
		# "typical_p": 0.7,
		# "repeat_last_n": 33,
		# "temperature": 0.8,
		# "repeat_penalty": 1.2,
		# "presence_penalty": 1.5,
		# "frequency_penalty": 1.0,
		# "mirostat": 1,
		# "mirostat_tau": 0.8,
		# "mirostat_eta": 0.6,
		# "penalize_newline": True,
		"stop": ["user:"],
		# "numa": False,
		"num_ctx": 16384, # 默认记忆大小
		# "num_batch": 2,
		# "num_gqa": 1,
		# "num_gpu": 1,
		# "main_gpu": 0,
		# "low_vram": False,
		# "f16_kv": True,
		# "vocab_only": False,
		# "use_mmap": True,
		# "use_mlock": False,
		# "embedding_only": False,
		# "rope_frequency_base": 1.1,
		# "rope_frequency_scale": 0.8,
		# "num_thread": 8
		}

	def todict(self) -> dict:
		"""
		获得一个默认的options类
		"""
		return self.dict
	
	def addstop(self, stop:list[str]) -> dict:
		"""
		添加一个stop参数到options中
		"""
		for word in stop:
			self.dict['stop'].append(word)
		return self.dict
	
	def replacestop(self, stop:list[str]) -> dict:
		"""
		替换掉stop中的所有参数
		"""
		self.dict['stop'] = stop
		return self.dict
	
	def setMaxToken(self, num:int) -> None:
		self.dict['num_predict'] = num

class OllamaMessages:
	"""
	创建一个messages类便于之后使用
	"""
	def __init__(self, prompt:str) -> None:
		self.messages = [
			{
				'role':'user',
				'content': prompt,
			}
		]
	
	def append(self, message:str) -> None:
		self.messages.append({
			'role':'user',
			'content': message,
		})

	def tolist(self) -> list:
		return self.messages


def creatOllamaRequest(messages:OllamaMessages, options:OllamaRequestOptions, model='llama2', isjson=False) :
	"""
	向ollama请求返回
	messages : 为一个列表,含有 'role','content'等属性
	options : 包含对于该请求的设置,如stop, maxtoken
	model : 默认使用llama2:7b来进行生成
	"""
	# 
	response = ollama.chat(model, messages.tolist(), stream=False, options=options.todict())
	# print(response['message']['content'])
	if isjson == True:
		ret = json.loads(ret)
		print(ret)
	ret = response['message']['content']

	return ret

if __name__ == '__main__':

	message = OllamaMessages("why is the sky blue?")
	option = OllamaRequestOptions()
	ret = creatOllamaRequest(message, option)
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