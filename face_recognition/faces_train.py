import os 
import cv2 as cv
import numpy as np


# metodo para criar lista utilizando os folders 
pessoas = []

for i in os.listdir(r'C:\Users\gabri\Desktop\cvton\face_recognition\jogadores'):
    pessoas.append(i)


# variavel com o local onde estarãoo as referencias 
    
DIR = r'C:\Users\gabri\Desktop\cvton\face_recognition\jogadores'

# pegar a classificação do haar cascade como base para analise dos rostos
haar_cascade = cv.CascadeClassifier("haar_face.xml")
features = []
labels = []

# criando a função que vai treinar o programa pra reconhecer cada rosto
def create_train():
    for pessoa in pessoas:
        # separando o cada folder como uma pessoa diferente
        path = os.path.join(DIR, pessoa)  
        # colocando um index em cada folder separando todos   
        label = pessoas.index(pessoa)
        
        # pegando as imagens dentro dos folder no lambreteiros
        for img in os.listdir(path):
            # relacionando a imagem com o arquivo em si
            img_path = os.path.join(path,img)
            # codigo para o programa ler as imagens dentro dos folders
            img_array = cv.imread(img_path)
            # transformando elas em grayscale por ele não ler cor e sim as caracteristicas do rosto
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            # codigo para a detecção dos rostos dentro das imagens 
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
            # codigo para buscar cada feature do rosto e colocar elas dentro das listas correspondendo a pessoa 
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
                
create_train()

print("O treino foi finalizado, agr to malando")

#print(f'o tamanho das features é de: {len(features)}')
#print(f'o tamanho dos labels é de: {len(labels)}')
# transformando as listas em arrays do numpy 
features = np.array(features, dtype="object")
labels = np.array(labels)

# criar a variavel com a função que reconhece os rostos nas imagens
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# aqui para treinar o programa com as features dos rostos e seu index
face_recognizer.train(features,labels)     
# salvando dentro de arquivos separados
face_recognizer.save("face_trained.yml")
np.save("features.npy", features)
np.save("labels.npy", labels)
            