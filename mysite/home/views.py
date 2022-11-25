from django.contrib.auth.models import User
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
import datetime

import numpy as np
import cv2
import sys
import time
from random import randint
from reportlab.pdfgen import canvas
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import shutil
import imutils


from . import validator


class Box:
    def __init__(self, start_point, width_height):
        self.start_point = start_point
        self.end_point = (start_point[0] + width_height[0], start_point[1] + width_height[1])
        self.counter = 0
        self.frame_countdown = 0

    def overlap(self, start_point, end_point):
        if self.start_point[0] >= end_point[0] or self.end_point[0] <= start_point[0] or \
                self.start_point[1] >= end_point[1] or self.end_point[1] <= start_point[1]:
            return False
        else:
            return True



from django.urls import reverse
# Create your views here.

TEMPLATE_DIRS = (
    'os.path.join(BASE_DIR, "templates"),'
)

def index(request):
    today = datetime.datetime.now()
    return render(request, "index.html", {"today": today})

def perform_login(request):
    today = datetime.datetime.now()
    if request.method != "POST":
        return HttpResponse("Metodo não aceitável")
    else:
            emailText = request.POST.get("emailText")
            senhaText = request.POST.get("senhaText")
            user_obj = authenticate(request, username = emailText, password = senhaText)

            if user_obj is not None:
                login(request, user_obj)
                return HttpResponseRedirect(reverse("minhas_analises"))
            else:
                return render(request, "index.html", {"today": today})

def admin_dashboard(request):
    return render(request, "admin_dashboard.html")

def perform_logout(request):
    logout(request)
    return HttpResponseRedirect("/")

def minhas_analises(request):
    return render(request, "minhas_analises.html")

def meus_documentos(request):
    return render(request, "meus_documentos.html")

def cadastro_usuario(request):
    return render(request,"cadastro_usuario.html")

def retorna_pdf(request):
    return render(request,"Detritos.pdf")

def cadastrar_usuario(request):
    today = datetime.datetime.now()
    try:
        usuario_aux = User.objects.get(email=request.POST['emailCad'])
        if usuario_aux:
            return render(request, 'cadastro_usuario.html', {'msg': 'Usuario existente',"today": today})

    except User.DoesNotExist:
        usuario = request.POST['usuarioCad']
        email = request.POST['emailCad']
        senha = request.POST['senhaCad']

        novoUsuario = User.objects.create_user(username=usuario, email=email, password=senha)
        novoUsuario.save()
        return HttpResponseRedirect(reverse("admin_dashboard"))







def getKernel(KERNEL_TYPE):
    if KERNEL_TYPE == "dilation":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if KERNEL_TYPE == "opening":
        kernel = np.ones((3, 3), np.uint8)
    if KERNEL_TYPE == "closing":
        kernel = np.ones((11, 11), np.uint8)

    return kernel

def getFilter(img, filter):
    if filter == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)

    if filter == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)

    if filter == 'dilation':
        return cv2.dilate(img, getKernel("dilation"), iterations=2)

    if filter == 'combine':
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, getKernel("closing"), iterations=2)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, getKernel("opening"), iterations=2)
        dilation = cv2.dilate(opening, getKernel("dilation"), iterations=2)

        return dilation

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=120, decisionThreshold=.8)
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG(history=200, nmixtures=5, backgroundRatio=.7, noiseSigma=0)
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=False, varThreshold=200)
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN(history=100, dist2Threshold=400, detectShadows=True)
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory=True,
                                                        maxPixelStability=15 * 60, isParallel=True)
    print("Detector inválido")
    sys.exit(1)

def getCentroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return (cx, cy)

def getDistance(area):
    #1cm = 37,80 px
    #Objeto de 10cm² a 1m de distância ocupa 5.714 px
    d = 5714/area
    return round(d, 2)


def video_tr():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    ret, frame = cap.read()
    # Segura o último frame para verificar se houve movimento
    last_frame = None
    # Texto que irá exibir o número de objetos
    text = ""
    # Caixas que irão contar os objetos
    boxes = []
    boxes.append(Box((100, 200), (10, 80)))
    boxes.append(Box((300, 350), (10, 80)))
    while cap.isOpened():
        _, frame = cap.read()
        # Processamento dos frames
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            break
        # Para minimizar os pequenos detalhes
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if last_frame is None or last_frame.shape != gray.shape:
            last_frame = gray
            continue
        # Pega a diferença entre o frame atual e o último
        delta_frame = cv2.absdiff(last_frame, gray)
        last_frame = gray
        # Threshold - Atribuição dos valores em pixels de acordo com o limite
        thresh = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)
        # Retorna uma lista de objetos
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Conversão
        contours = imutils.grab_contours(contours)
        # Looping em todos os objetos
        for contour in contours:
            # Pula se a diferença for pequena (pode ser ajustado)
            if cv2.contourArea(contour) < 500:
                continue
            # Destaca o objeto colocando as caixas
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            text = "Objetos:"
            # Passa por todas as caixas
            for box in boxes:
                box.frame_countdown -= 1
                if box.overlap((x, y), (x + w, y + h)):
                    if box.frame_countdown <= 0:
                        box.counter += 1
                    # The number might be adjusted, it is just set based on my settings
                    box.frame_countdown = 20
                text += " (" + str(box.counter) + " ," + str(box.frame_countdown) + ")"
        # Insere o texto da quantidade de elementos
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        # Insere as caixas
        for box in boxes:
            cv2.rectangle(frame, box.start_point, box.end_point, (255, 255, 255), 2)
        # Abre a janela
        cv2.imshow("Identificacao de objetos", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def GeneratePDF(lista):
    try:
        nome_pdf = "Detritos"
        pdf = canvas.Canvas('{}.pdf'.format(nome_pdf))
        x = 720
        for nome, idade in lista.items():
            x -= 20
            pdf.drawString(30, x, '{} : {}'.format(nome, idade))
        pdf.setTitle(nome_pdf)
        pdf.setFont("Helvetica-Oblique", 14)
        pdf.drawString(30, 750, 'Possíveis detritos identificados')
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(30, 724, 'ID do Objeto - Horário de Identificação - Distância aproximada (M)')
        pdf.save()
        print('{}.pdf criado com sucesso!'.format(nome_pdf))
    except:
        print('Erro ao gerar {}.pdf'.format(nome_pdf))

#print(getCentroid(50, 100, 100, 100))

def save_frame(frame, file_name, flip=True):
    if flip: # BGR -> RGB
        cv2.imwrite(file_name, np.flip(frame, 2))
    else:
        cv2.imwrite(file_name, frame)


def main(request):
    LINE_IN_COLOR = (64, 255, 0)
    LINE_OUT_COLOR = (0, 0, 255)
    BOUNDING_BOX_COLOR = (255, 128, 0)
    TRACKER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
    CENTROID_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
    TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
    TEXT_POSITION_BGS = (10, 50)
    TEXT_POSITION_COUNT_CARS = (10, 100)
    TEXT_POSITION_COUNT_TRUCKS = (10, 150)
    TEXT_SIZE = 1.2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SAVE_IMAGE = True
    IMAGE_DIR = "C:\\Users\\Lucas\\PycharmProjects\\TesteDjango\\mysite\\home\\imagens"
    #VIDEO_SOURCE = "C:\\Users\\Lucas\\PycharmProjects\\debrisIdentification\\videos\\Traffic_3.mp4"
    VIDEO_SOURCE = request.POST.get("urlArquivo")
    VIDEO_OUT = "videos/results/result_traffic.avi"
    totdeb = 0
    dstmed = 0

    BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]
    BGS_TYPE = BGS_TYPES[2]

    frame_number = -1
    cnt_cars, cnt_trucks = 0, 0
    objects = []
    max_p_age = 2
    pid = 1
    lista = {}

    print(VIDEO_SOURCE)
    if(VIDEO_SOURCE == "0"):
        video_tr()
    else:


        cap = cv2.VideoCapture(VIDEO_SOURCE)
        hasFrame, frame = cap.read()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # writer_video = cv2.VideoWriter(VIDEO_OUT, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

        # ROI
        bbox = cv2.selectROI(frame, False)
        print(bbox)

        (w1, h1, w2, h2) = bbox

        frameArea = h2 * w2
        # print(frameArea)
        minArea = int(frameArea / 250)
        maxArea = 15000
        # print(minArea)

        line_IN = int(h1)
        line_OUT = int(h2 - 20)
        # print(line_IN, line_OUT)

        DOWN_limit = int(h1 / 4)
        print('Down IN limit Y', str(DOWN_limit))
        print('Down OUT limit Y', str(line_OUT))

        bg_subtractor = getBGSubtractor(BGS_TYPE)

        while (cap.isOpened()):

            ok, frame = cap.read()
            if not ok:
                print("Erro")
                break

            roi = frame[h1:h1 + h2, w1:w1 + w2]

            for i in objects:
                #print('teste')
                i.age_one()

            frame_number += 1
            bg_mask = bg_subtractor.apply(roi)
            bg_mask = getFilter(bg_mask, 'combine')
            (contours, hierarchy) = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)

                if area > minArea and area <= maxArea:
                    x,y,w,h = cv2.boundingRect(cnt)
                    centroid = getCentroid(x, y, w, h)
                    cx = centroid[0]
                    cy = centroid[1]
                    new = True
                    cv2.rectangle(roi, (x, y), (x + 50, y - 13), TRACKER_COLOR, -1)
                    cv2.putText(roi, 'OBJ', (x, y - 2), FONT, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    for i in objects:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)

                            if i.going_DOWN(DOWN_limit) == True:
                                cnt_cars += 1
                                if SAVE_IMAGE:
                                    save_frame(roi, IMAGE_DIR + '/obj_DOWN_%04d.png' % frame_number)
                                    id_deb = i.getId()
                                    hor_deb = time.strftime("%c")
                                    dist_deb = getDistance(area)
                                    area_deb = str(area)
                                    info_deb = hor_deb + " --- " + str(dist_deb)
                                    print("ID:", id_deb, ' detectado em ', hor_deb, ' - Área em px: ', area_deb)
                                    lista[id_deb] = info_deb
                                    totdeb += 1
                                    dstmed += dist_deb
                            break
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > line_OUT:
                                i.setDone()
                        if i.timedOut():
                            index = objects.index(i)
                            objects.pop(index)
                            del i
                    if new == True:
                        p = validator.MyValidator(pid, cx, cy, max_p_age)
                        objects.append(p)
                        pid += 1
                    cv2.circle(roi, (cx, cy), 5, CENTROID_COLOR, -1)

                # Cc
                elif area >= maxArea:
                    x, y, w, h = cv2.boundingRect(cnt)
                    centroid = getCentroid(x, y, w, h)
                    cx = centroid[0]
                    cy = centroid[1]

                    new = True

                    cv2.rectangle(roi, (x, y), (x + 50, y - 13), TRACKER_COLOR, -1)
                    cv2.putText(roi, 'OBJ*', (x, y - 2), FONT, .5, (255, 255, 255), 1, cv2.LINE_AA)

                    for i in objects:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)

                            if i.going_DOWN(DOWN_limit) == True:
                                cnt_trucks += 1
                                if SAVE_IMAGE:
                                    save_frame(roi, IMAGE_DIR + '/obj_DOWN_%04d.png' % frame_number)
                                    id_deb = i.getId()
                                    hor_deb = time.strftime("%c")
                                    dist_deb = getDistance(area)
                                    area_deb = str(area)
                                    info_deb = hor_deb + " --- " + str(dist_deb)
                                    print("ID:", id_deb, ' detectado em ', hor_deb, ' - Área em px: ', area_deb)
                                    lista[id_deb] = info_deb
                                    totdeb += 1
                                    dstmed += dist_deb
                            break
                        if i.getState() == '1':
                            if i.getDir() == 'down' and i.getY() > line_OUT:
                                i.setDone()
                        if i.timedOut():
                            index = objects.index(i)
                            objects.pop(index)
                            del i
                    if new == True:
                        p = validator.MyValidator(pid, cx, cy, max_p_age)
                        objects.append(p)
                        pid += 1
                    cv2.circle(roi, (cx, cy), 5, CENTROID_COLOR, -1)

            for i in objects:
                cv2.putText(roi, str(i.getId()), (i.getX(), i.getY()), FONT, 0.3, TEXT_COLOR, 1, cv2.LINE_AA)

            str_cars = 'Objetos: ' + str(cnt_cars)
            str_trucks = 'Objetos*G: ' + str(cnt_trucks)

            frame = cv2.line(frame, (w1, line_IN), (w1 + w2, line_IN), LINE_IN_COLOR, 2)
            frame = cv2.line(frame, (w1, h1 + line_OUT), (w1 + w2, h1 + line_OUT), LINE_OUT_COLOR, 2)

            cv2.putText(frame, str_cars, TEXT_POSITION_COUNT_CARS, FONT, 1, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(frame, str_cars, TEXT_POSITION_COUNT_CARS, FONT, 1, (232, 162, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, str_trucks, TEXT_POSITION_COUNT_TRUCKS, FONT, 1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, str_trucks, TEXT_POSITION_COUNT_TRUCKS, FONT, 1, (232, 162, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, 'Identificar Detritos: ' + BGS_TYPE, TEXT_POSITION_BGS, FONT, TEXT_SIZE, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, 'Identificar Detritos: ' + BGS_TYPE, TEXT_POSITION_BGS, FONT, TEXT_SIZE, (128, 0, 255), 2, cv2.LINE_AA)

            for alpha in np.arange(0.3, 1.1, 0.9)[::-1]:
                overlay = frame.copy()
                output = frame.copy()
                cv2.rectangle(overlay, (w1, h1), (w1 + w2, h1 + h2), BOUNDING_BOX_COLOR, -1)
                frame = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', bg_mask)

            #writer_video.write(frame)


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


        cap.release()
        cv2.destroyAllWindows()

        GeneratePDF(lista)

        shutil.make_archive('imagens', 'zip', './', IMAGE_DIR)

        fromaddr = "reconhecimentodetritos@outlook.com"
        toaddr = "pinheiro.lucas@aluno.ifsp.edu.br"

        msg = MIMEMultipart()
        msg['From'] = fromaddr
        msg['To'] = toaddr
        msg['Subject'] = "Identificação de Detritos"
        body = "Prezado(a), segue documento gerado pela análise do vídeo."
        msg.attach(MIMEText(body, 'plain'))
        filename = "Detritos.pdf"
        attachment = open("Detritos.pdf", "rb")
        p = MIMEBase('application', 'octet-stream')
        p.set_payload((attachment).read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
        msg.attach(p)

        filename2 = "Imagens.zip"
        attachment2 = open("imagens.zip", "rb")
        p2 = MIMEBase('application', 'octet-stream')
        p2.set_payload((attachment2).read())
        encoders.encode_base64(p2)
        p2.add_header('Content-Disposition', "attachment; filename= %s" % filename2)
        msg.attach(p2)

        s = smtplib.SMTP('smtp.office365.com', 587)
        s.starttls()
        s.login(fromaddr, "Detritos@2022")
        text = msg.as_string()
        s.sendmail(fromaddr, toaddr, text)
        s.quit()

    if(totdeb != 0):
        dstmed = dstmed/totdeb

    dstmed = round(dstmed, 2)
    totdebstr = "Total de Detritos encontrados: " + str(totdeb)
    dstmedstr = "Distância média dos detritos: " + str(dstmed) + " metros"
    return render(request, "minhas_analises.html", {"totdebstr": totdebstr, "dstmedstr": dstmedstr})



