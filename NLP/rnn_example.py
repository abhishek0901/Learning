import torch
from torch import nn
import numpy as np

text = ['hey how are you',
        'good i am fine',
        'have a nice day',
        'somethings are damn beautiful']

#Data preprocessing and converting char/words to vectors/ints
chars = set(''.join(text)) #Get unique chars
int2char = dict(enumerate(chars)) #int to chars
char2int = {char: ind for ind, char in int2char.items()} #char to ints

maxlen = len(max(text,key=len)) #Gives length of maximum sentence

#Padding
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '

input_seq = []
target_seq = []

for i in range(len(text)):
	input_seq.append(text[i][:-1])
	target_seq.append(text[i][1:])

for i in range(len(text)):
	input_seq[i] = [char2int[chracter] for chracter in input_seq[i]]
	target_seq[i] = [char2int[chracter] for chracter in target_seq[i]]

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence,dict_size,seq_len,batch_size):
	features = np.zeros((batch_size,seq_len,dict_size))
	for i in range(batch_size):
		for u in range(seq_len):
			features[i,u,sequence[i][u]] = 1

	return features

input_seq = one_hot_encode(input_seq,dict_size,seq_len,batch_size)

input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)


is_cuda = torch.cuda.is_available()
if is_cuda:
	device = torch.device('cuda')
else:
	device = torch.device('cpu')



class RNNModel(nn.Module):
	"""docstring for RNNModel"""
	def __init__(self, input_size,output_size,hidden_dim,n_layers):
		super(RNNModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers

		#RNN Layer
		self.rnn = nn.RNN(input_size,hidden_dim,n_layers,batch_first=True)

		#Fully Connected Layer
		self.fc = nn.Linear(hidden_dim,output_size)

	def forward(self,x):
		batch_size = x.size(0)
		hidden = self.init_hidden(batch_size)
		out, hidden = self.rnn(x,hidden)
		out = out.contiguous().view(-1,self.hidden_dim)
		out = self.fc(out)

		return out,hidden

	def init_hidden(self,batch_size):
		hidden = torch.zeros(self.n_layers,batch_size,self.hidden_dim)
		return hidden

model = RNNModel(input_size=dict_size,output_size=dict_size,hidden_dim = 12,n_layers=1).to(device)
#model.to(device)

n_epochs = 1000
lr = 0.001


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = lr)



for epochs in range(1,n_epochs + 1):
	optimizer.zero_grad()
	input_seq = input_seq.to(device).type(torch.FloatTensor)
	output, hidden = model(input_seq)
	loss = criterion(output,target_seq.view(-1).long())
	loss.backward()
	optimizer.step()

	if epochs % 100 == 0:
		print(f'Epoch: {epochs}/{n_epochs} ...................... Loss: {loss.item()}')

def predict(model,character):
	character = np.array([[char2int[c] for c in character]])
	character = one_hot_encode(character,dict_size,character.shape[0],1)
	character = torch.from_numpy(character)
	character = character.to(device).type(torch.FloatTensor)

	out,hidden = model(character)

	prob = nn.functional.softmax(out[-1],dim = 0).data
	char_ind = torch.max(prob,dim=0)[1].item()

	return int2char[char_ind], hidden

def sample(model,out_len,start = 'hey'):
	model.eval()
	start = start.lower()
	chars = [ch for ch in start]
	size = out_len - len(chars)

	for ii in range(size):
		char,h = predict(model,chars)
		chars.append(char)

	return ''.join(chars)

print(sample(model,15,'beautiful'))
