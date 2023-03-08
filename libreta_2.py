import warnings
warnings.filterwarnings('ignore')


from collections import defaultdict
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.lm.preprocessing import flatten

import re

def Leer_corpus(path,nrows=-1):
    corpus_raw	= pd.DataFrame(columns=['sentencias'])
    if nrows > 0 :
        with open(path,'r',encoding="utf-8") as frame:
            contador = 0;
            for linea in frame:
                if contador == nrows :
                    break
                auxiliar = pd.DataFrame( [linea],columns=['sentencias'])
                corpus_raw = pd.concat( [corpus_raw, auxiliar ] ,ignore_index=True)
                contador=contador+1					
    else:
        with open(path,'r',encoding="utf-8") as frame:
            for linea in frame:
                auxiliar = pd.DataFrame( [linea],columns=['sentencias'])
                corpus_raw = pd.concat( [corpus_raw, auxiliar ] ,ignore_index=True)
    return corpus_raw

path ="./conferencias_matutinas_amlo.txt"
corpus_crudo = Leer_corpus(path)

print("corpus en memoria...")

def todo_UNK(sentencia):
        for palabra in sentencia:
            if(palabra != '<UNK>'):
                return False
        return True

def Procesamiento_invasivo (corpus):

    # <- Una lista de listas(sublistas). Cada sublistas tiene solo un elemento.
    corpus_preprocesado = corpus_crudo.values.tolist()     

    #Se eliminan las sentencias que tengan numero. 
    corpus_preprocesado = [ sentencia if (re.search(r'\d+',sentencia[0]) == None) else []   
                           for sentencia in corpus_preprocesado]
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]


    #Se transforma a lista de listas y se aplica el lower
    # NOTA: el lower tambien funiona para palabras con acentos 
    corpus_preprocesado = [ re.sub(r'\W',' ',sentencia[0]).lower()  
                           for sentencia in corpus_preprocesado]


    #Se quitan los acentos
    vocales = (
        ('á' , 'a'),
        ('é', 'e'),
        ('í' , 'i'),
        ('ó' , 'o'),
        ('ú' , 'u')
    )
    for vocales_acentuadas,vocales_regulares in vocales:
        corpus_preprocesado = [ sentencia.replace(vocales_acentuadas,vocales_regulares) 
                           for sentencia in corpus_preprocesado]
        

    #Se separan las letras 
    corpus_preprocesado = [ sentencia.split()  
                           for sentencia in corpus_preprocesado]
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]



    #Se Omiten las sentencias que tengan 2 o menos palabras
    #   Se aprovecha para remover tambien las sentencias que en primera intancia no tenian nada
    corpus_preprocesado = [ (sentencia if len(sentencia) >= 3 else [])   
                           for sentencia in corpus_preprocesado]
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]
    
    
    

    #A las sentencias muy largas se les quita algunas palabras del inicio y otras del final
    # Puede que este procesamiento no tenga mucho sentido. es mas un ejercicio de regex 
    # que algo ultil
    corpus_preprocesado = [ (sentencia if len(sentencia) <= 35 else sentencia[5:-6])   
                           for sentencia in corpus_preprocesado]
    
    
    #Para los siguientes preprocesamientos, se calcula la frecuencia de cada palabra
    palabras_frecuentes = Counter(flatten(corpus_preprocesado))

        
    #Se cambian las palabras poco frecuentes en el corpus
    #   Para quitar los n elementos menos frecuentes es frecuencias.most_frecuens()[ : -1 - n , -1 ]
    for index in range(len(corpus_preprocesado)):
        corpus_preprocesado[index] = [ palabra if 
                        palabras_frecuentes[palabra] > 3
                        else '<UNK>'  
                        for palabra in corpus_preprocesado[index]]
    
    #Si una sentencia tiene puras palabras UNK, se elimina
    corpus_preprocesado = [  sentencia if not todo_UNK(sentencia) else []  for sentencia in corpus_preprocesado]    
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]
    
    return corpus_preprocesado

def Procesamiento_prudente (corpus):

    # <- Una lista de listas(sublistas). Cada sublistas tiene solo un elemento.
    corpus_preprocesado = corpus_crudo.values.tolist()     



    #Se transforma a lista de listas y se aplica el lower
    # NOTA: el lower tambien funiona para palabras con acentos 
    corpus_preprocesado = [ re.sub(r'\W',' ',sentencia[0]).lower()  
                           for sentencia in corpus_preprocesado]

    #Cambiamos los numeros por un token
    corpus_preprocesado = [ re.sub(r'\d+','<DGT>',sentencia)  
                           for sentencia in corpus_preprocesado]
    

    #Se quitan los acentos
    vocales = (
        ('á' , 'a'),
        ('é', 'e'),
        ('í' , 'i'),
        ('ó' , 'o'),
        ('ú' , 'u')
    )
    for vocales_acentuadas,vocales_regulares in vocales:
        corpus_preprocesado = [ sentencia.replace(vocales_acentuadas,vocales_regulares) 
                           for sentencia in corpus_preprocesado]
        

    #Se separan las letras 
    corpus_preprocesado = [ sentencia.split()  
                           for sentencia in corpus_preprocesado]
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]



    #Se Omiten las sentencias que tengan 2 o menos palabras
    #   Se aprovecha para remover tambien las sentencias que en primera intancia no tenian nada
    corpus_preprocesado = [ (sentencia if len(sentencia) >= 1 else [])   
                           for sentencia in corpus_preprocesado]
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]
    
    
    
    #Para los siguientes preprocesamientos, se calcula la frecuencia de cada palabra
    palabras_frecuentes = Counter(flatten(corpus_preprocesado))
    #Se cambian las palabras poco frecuentes en el corpus
    #   Para quitar los n elementos menos frecuentes es frecuencias.most_frecuens()[ : -1 - n , -1 ]
    for index in range(len(corpus_preprocesado)):
        corpus_preprocesado[index] = [ palabra if 
                        palabras_frecuentes[palabra] > 2
                        else '<UNK>'  
                        for palabra in corpus_preprocesado[index]]
    
    #Si una sentencia tiene puras palabras UNK, se elimina
    corpus_preprocesado = [  sentencia if not todo_UNK(sentencia) else []  for sentencia in corpus_preprocesado]    
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]
    
    return corpus_preprocesado
    
def Procesamiento_simple(corpus):

    # <- Una lista de listas(sublistas). Cada sublistas tiene solo un elemento.
    corpus_preprocesado = corpus_crudo.values.tolist()     


    #Se transforma a lista de listas y se aplica el lower
    # NOTA: el lower tambien funiona para palabras con acentos 
    corpus_preprocesado = [ re.sub(r'\W',' ',sentencia[0])  
                           for sentencia in corpus_preprocesado]
    
    
    #Se separan las letras y se eliminan los sentencas vacias
    corpus_preprocesado = [ sentencia.split()  
                           for sentencia in corpus_preprocesado]
    corpus_preprocesado = [ sentencia for sentencia in corpus_preprocesado if sentencia != []]    

    
    #Se cambian las palabras poco frecuentes en el corpus por el token <UNK>
    palabras_frecuentes = Counter(flatten(corpus_preprocesado))
    for index in range(len(corpus_preprocesado)):
        corpus_preprocesado[index] = [ palabra if 
                        palabras_frecuentes[palabra] > 1
                        else '<UNK>'  
                        for palabra in corpus_preprocesado[index]]

    return corpus_preprocesado

print("Empieza procesamiento...")
corpus_procesamiento_prudente= Procesamiento_prudente(corpus_crudo)
corpus_procesamiento_simple = Procesamiento_simple(corpus_crudo)
corpus_procesamiento_invasivo = Procesamiento_invasivo(corpus_crudo)
print("Termina procesamiento")

'''
class EDA_PLN():
	
	#El objeto obtiene el corpus en el formato lista de listas [ ['token' , 'token' ... 'token'] , ['token',token ...] , ...]
	def __init__(self,corpus) -> None:
		self.corpus	= corpus
		self.corpus_unidimensional = []
		for sentencia in self.corpus:
			for token in sentencia:
				self.corpus_unidimensional.append(token)

	#Toma el corpus y regresa la frecuencia de los n_grama
	def get_top_ngram(self, n_grams=1):
		gramas = (ngrams(self.corpus_unidimensional,n_grams))
		frecuencias = FreqDist( gramas  )
		return frecuencias.most_common(10)
		
	#Unica funcion que si funciona :D Tomen esta como ejemplo
	def Caracteres_por_sentencia(self,
			    eje = None,
			    tamano_titulo = 18,
			    tamano_ejes = 10,
			    titulo="Histograma de longitud por sentencias"):
		if eje is None:	
			presentacion ,eje = plt.subplots(1)


		eje.hist( list(map( lambda x : len(x)  , self.corpus )) )
		
		eje.set_title(titulo, fontsize = tamano_titulo)
		eje.set_ylabel("Frecuencia", fontsize = tamano_ejes)
		eje.set_xlabel("Longitud",fontsize = tamano_ejes)

	#Copia de 'caracteres_por_sentencia'
	def Palabras_por_sentencia(self,
			    eje = None,
			    tamano_titulo = 18,
			    tamano_ejes = 10,
			    titulo="Histograma de longitud por sentencias"):
		if eje is None:	
			presentacion ,eje = plt.subplots(1)


		eje.hist( list(map( lambda x : len(x)  , self.corpus )) )
		
		eje.set_title(titulo, fontsize = tamano_titulo)
		eje.set_ylabel("Frecuencia", fontsize = tamano_ejes)
		eje.set_xlabel("Longitud",fontsize = tamano_ejes)	

	#Depregated, no esta funcionando, lo fixeo despues
	def Longitud_promedio_de_palabras(self,
			    eje = None,
			    tamano_titulo = 18,
			    tamano_ejes = 10,
			    titulo="Longitud promedio de palabras por sentencias"):
		if eje is None:	
			presentacion ,eje = plt.subplots(1)

		eje.hist(self.corpus['sentencias'].str.split().apply(
			lambda x : [len(i) for i in x]).map(
			lambda x: np.mean(x)))
		
		eje.set_title(titulo, fontsize = tamano_titulo)
		eje.set_ylabel("Frecuencia", fontsize = tamano_ejes)
		eje.set_xlabel("Longitud",fontsize = tamano_ejes)

	#Depregated, no esta funcionando, lo fixeo despues
	def Palabras_mas_comunes(self,
			  	stopwords = {''},
			    eje = None,
			    tamano_titulo = 18,
			    tamano_ejes = 10,
			    titulo="Palabras mas comunes"):
		if eje is None:	
			presentacion ,eje = plt.subplots(1)
		
		palabras_relevantes = defaultdict(int)
		for key in self.frecuencias :
			if key not in stopwords:
				palabras_relevantes[key] = self.frecuencias[key]

		top=sorted(palabras_relevantes.items(), key=lambda x:x[1],reverse=True)[:10] 
		x,y=zip(*top)
		eje.bar(x,y)
		
		eje.set_title(titulo, fontsize = tamano_titulo)
		eje.set_ylabel("Frecuencia", fontsize = tamano_ejes)
		eje.set_xlabel("Longitud",fontsize = tamano_ejes)

	#Depregated, no esta funcionando, lo fixeo despues
	def Stopwords_mas_comunes(self,
			  	stopwords,
			    eje = None,
			    tamano_titulo = 18,
			    tamano_ejes = 10,
			    titulo="Stopwords mas comunes"):
		if eje is None:	
			presentacion ,eje = plt.subplots(1)
		
		palabras_relevantes = defaultdict(int)
		for key in self.frecuencias :
			if key in stopwords:
				palabras_relevantes[key] = self.frecuencias[key]

		top=sorted(palabras_relevantes.items(), key=lambda x:x[1],reverse=True)[:10] 
		x,y=zip(*top)
		eje.bar(x,y)
		
		eje.set_title(titulo, fontsize = tamano_titulo)
		eje.set_ylabel("Frecuencia", fontsize = tamano_ejes)
		eje.set_xlabel("Longitud",fontsize = tamano_ejes)
'''

print("Empieza guardado en memoria")
import random
def split_file(file, out1, out2, percentage=0.75, isShuffle=True, seed=764):
    """Splits a file in 2 given the `percentage` to go in the large file."""
    random.seed(seed)
    with open(file, 'r',encoding="utf-8") as file, \
         open(out1, 'w', encoding="utf-8") as foutBig, \
         open(out2, 'w', encoding="utf-8") as foutSmall:
                nLines = sum(1 for line in file)
                file.seek(0)
                nTrain = int(nLines*percentage) 
                nValid = nLines - nTrain
                i = 0
                for line in file:
                    r = random.random() if isShuffle else 0 # so that always evaluated to true when not isShuffle
                    if (i < nTrain and r < percentage) or (nLines - i > nValid):
                        foutBig.write(line)
                        i += 1
                    else:
                        foutSmall.write(line)
                    
def corpus_procesato_a_csv(corpus,nombre_csv):
    corpus_pd = pd.DataFrame({'sentencias' :corpus}  )
    corpus_pd.to_csv(nombre_csv , index = False)
    split_file(nombre_csv,nombre_csv+'_entrenamiento',nombre_csv+'_prueba')

corpus_procesato_a_csv(corpus_procesamiento_simple,"procesado_simple")
corpus_procesato_a_csv(corpus_procesamiento_prudente,"procesado_prudente")
corpus_procesato_a_csv(corpus_procesamiento_invasivo,"procesado_invasivo")

print("Fin Ejecucion")