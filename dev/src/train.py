import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Defining our model and de decoder function
class ProposedModel(nn.Module):

    def __init__(self, vocab_size , embedding_dim = 128, hidden_size = 500, num_layers = 1,context_size = 5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.linear = nn.Linear(context_size * hidden_size, vocab_size)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first= True)

        self.context_size = context_size
        self.embedding_dim = embedding_dim

    def forward(self, x, hidden = None):
        # Ref: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
        x = self.embedding_layer(x)

        output, hidden= self.gru(x, hidden)

        batch_size, sequence_len, hidden_size = output.shape

        output = output.contiguous().view(batch_size, sequence_len*hidden_size)

        output = F.dropout(output, 0.3)

        x = self.linear(output)
        # x = torch.relu(x)
        
        # x = classifier(x)
        return x
    
    def embedding(self, x, hidden = None):
        x = self.embedding_layer(x)

        output, hidden= self.gru(x, hidden)
        
        return output

# Defining our model and de decoder function
class BengioLModel(nn.Module):

    def __init__(self, vocab_size , embedding_dim = 128, hidden_size = 500, context_size = 5):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

        self.context_size = context_size
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # Ref: https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
        x = self.embedding_layer(x).view(-1,self.context_size*self.embedding_dim)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        
        return x
    
    def embedding(self, x):
        x = self.embedding_layer(x).view(-1,self.context_size*self.embedding_dim)
        x = self.linear1(x)
        
        return x
    
def decode_model(logits, k = 10, p = 0.95, method='greedy'):
    
    if(method=='greedy'):
        
        selected_tokens_ids = torch.argmax(logits).unsqueeze(dim=0).unsqueeze(dim=0)
        
    elif(method=='topk'):
        # Pegamos os valores e índices do top-10 logitos
        values, idxs = logits.topk(k)

        # Selecionamos um elemento aleatório de cada elemento do batch com o multinomial + softmax
        selected_idx = F.softmax(values, dim=-1).type(torch.float).multinomial(1)

        # Aplicamos o vetor de índices no vetor de token_ids e geramos os tokens selecionados
        selected_tokens_ids = idxs.gather(-1, selected_idx)
    elif(method=='ns'):
        # Ordenamos os logitos em ordem decrescente
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # Geramos probabilidade acumulada pois dai é mais fácil selecionar o ponto de corte (.95)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Gera vetor auxiliar que guardará os tokens ids
        selected_tokens_ids = torch.zeros((1, 1), dtype=torch.long, device=logits.device)
        # Para cada elemento do batch
        
        # Selecionamos os logitos onde o vetor acumulado ainda é menor que 0.95
        filtered = sorted_logits[cumulative_probs<p]
        # Se não tiver nenhum logito com cumprob <0.95, retornamos aquele com o logito mais alto
        if(len(filtered) == 0):
            selected_tokens_ids[0] = sorted_indices[0].unsqueeze(dim=0)
        # Se não for vazio, então:
        else:
            # Aplicamos softmax nos logitos filtrados
            filtered = F.softmax(filtered, dim = -1)
            # Selecionamos os tokens ids selecionados, amostrando um dos logitos filtrados e selecionando o mesmo índice
            # dos tokens ids ordenados
#             print(filtered, sorted_indices)
            selected_tokens_ids = sorted_indices[cumulative_probs<p].gather(-1, F.softmax(filtered, dim=-1).multinomial(1)).unsqueeze(dim=0)
        
    return selected_tokens_ids


# Definindo função de treino da rede
def train(train_loader, val_loader, model, epoch, optimizer, criterion, device, vocab, verbose = True):
  """
  Função que efetua o treinamento da rede. ref: https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e

  train_loader: torch df loader de treino
  val_loader: torch df loader de validação
  model: objeto da rede
  epoch: inteiro de quantidade de epocas a rodar
  optimizer: otimizador que será utilizado
  criterion: loss function que será utilizado
  verbose: bool para plotar avanço da rede ou nao

  return train_loss, val_loss, train_perplexity, val_perplexity
  
  """

  # Definindo listas de returns
  train_loss = []
  val_loss = []
  train_perplexity = []
  val_perplexity = []

  # Inicio do treinamento
  model.train()

  #

  for e in range(epoch):
    # Definindo os acúmulos de loss e perplexity a serem consolidados ao fim do treinamento em batchs
    local_train_loss = 0
    local_val_loss = 0
    local_train_ppl = 0
    local_val_ppl = 0

      ####
      ####        TREINAMENTO DO MODELO
      ####

    # Setando modo de treinamento
    model.train()

    # Normalizador servirá para pegar a média das loss
    normalizador = 1

    # Começo do treinamento em mini batches
    for x, y in train_loader:
      # Habilitando gpu 
      x = x.to(device)
      y = y.to(device)

      # Foward
      y_preds = model(x)

      # Loss
      loss = criterion(y_preds, y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Acúmulo de loss e ppl no treino
      local_train_loss += loss.item()

      # Incremento normalizador
      normalizador += 1

    # Fim treinamento, vamos consolidar as métricas
    mean_train_loss = local_train_loss/normalizador
    train_loss.append(mean_train_loss)
    train_perplexity.append(np.exp(mean_train_loss))

      ####
      ####        AVALIACAO DO MODELO
      ####

    # Começo de avaliação na validação

    # Setando modo de avaliação
    model.eval()

    normalizador = 1

    with torch.no_grad():
      # Começo da avaliação em mini batches
      for x, y in val_loader:
        # Habilitando gpu 
        x = x.to(device)
        y = y.to(device)

        # Foward
        y_preds = model(x)

        # Estressando <pad> e <unk> com o -1
        y_preds[:,vocab['<pad>']] = -1
        y_preds[:,vocab['<unk>']] = -1

        # Loss
        loss = criterion(y_preds, y)

        # Acúmulo de loss 
        local_val_loss += loss.item()

        normalizador += 1

    # Fim da avaliação, vamos consolidar as métricas
    mean_val_loss = local_val_loss/normalizador
    val_loss.append(mean_val_loss)
    val_perplexity.append(np.exp(mean_val_loss))

      ####
      ####        PRINT DE PROGRESSO CASO HABILITADO
      ####

    if(verbose):
      print(f"EPOCH {e+1}/{epoch}: trainning loss: {mean_train_loss} val loss: {mean_val_loss} | trainning ppl: {np.exp(mean_train_loss)} val ppl: {np.exp(mean_val_loss)}")

  # Return histórico
  return train_loss, val_loss, train_perplexity, val_perplexity

