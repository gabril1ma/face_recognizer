import numpy as np
import cv2 as cv 
import os

pessoas = []

for i in os.listdir(r'C:\Users\gabri\Desktop\cvton\face_recognition\jogadores'):
    pessoas.append(i)
    
haar_cascade = cv.CascadeClassifier("haar_face.xml")

# features = np.load("features.npy", allow_pickle=True)
# labels = np.load("labels.npy", allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

# aqui a imagem que vai ser lida pelo programa para analisar quem é 
person = cv.imread(r'C:\Users\gabri\Desktop\cvton\face_recognition\testes\ney (3).jpg')

gray = cv.cvtColor(person, cv.COLOR_BGR2GRAY)
# cv.imshow("person", gray)

# criando a variavel com as caracteristicas
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 8)
# o for que vai analisar todas as caracteristicas dentro da variavel
for (x,y,w,h) in faces_rect:
    # codigo padrão para analise das caracteristicas faciais
    faces_roi = gray[y:y+h,x:x+h]
    # colocanod o recognizer com o predict das caracteristicas faciais
    label, confidence = face_recognizer.predict(faces_roi)
    # aparecer no terminal a porcentagem de certeza que o programa teve de acertar a pessoa
    print(f'label = {pessoas[label]} with a precision of {round(confidence)}%')
    # colocar o texto com o nome da pessoa que 
    cv.putText(person, str(pessoas[label]), (20,50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    # coloca o retangulo envolta do rosto da pessoa
    cv.rectangle(person, (x,y), (x+w,y+h), (0,0,255), thickness=2)
    

cv.imshow("cara detectada", person)
cv.waitKey(0)