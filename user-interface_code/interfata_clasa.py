'''
/****************************************************************************
 *                                                                          *
 *  File:        cod_interfata.py                                           *
 *  Copyright:   (c) 2020, Maria Frentescu                                  *
 *  Description: This script is used to display an user interface where an  *
 *               user can upload images and make real-time predictions      *
 *               for age and gender.                                        *
 *                                                                          *
 ***************************************************************************/
'''
#-----imports-----
import cv2
from tkinter import *
from tkinter import filedialog,messagebox
from PIL import ImageTk
from PIL import Image
import torch
from torch.autograd import Variable
from ResNet50_age import ResNet_age
from ResNet50_gender import ResNet_gen
from VGG16_gender import Model_gen
from VGG16_age import Model_age
from torchvision import transforms
from os.path import isfile,join
import os

#-----globals-----
root = Tk()
root.title('Age and gender prediction')
root.geometry("1300x750")

full_image = ""
message = ""
image_vector = [""]
# this folder is used to save the extracted face images from input photo
# replace the path with your folder
prediction_path = "D:/Frentescu_Maria_cod_sursa/cod_interfata/poze_predictie/"

# device can be cpu or cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models_flag = 0
number_person = 0
label_flag = 0
name_flag = 0
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#init model
model_age_vgg = Model_age()
model_gen_vgg = Model_gen()
model_age_res = ResNet_age(img_channel=3, num_classes=1)
model_gen_res = ResNet_gen(img_channel=3, num_classes=2)

#load models
model_age_vgg.load_state_dict(torch.load("D:/Frentescu_Maria_cod_sursa/modele_antrenate/model_vgg16_age.pt", map_location=device))
model_gen_vgg.load_state_dict(torch.load("D:/Frentescu_Maria_cod_sursa/modele_antrenate/model_vgg16_gender.pt", map_location=device))
model_age_res.load_state_dict(torch.load("D:/Frentescu_Maria_cod_sursa/modele_antrenate/model_resnet50_age.pt", map_location=device))
model_gen_res.load_state_dict(torch.load("D:/Frentescu_Maria_cod_sursa/modele_antrenate/model_resnet50_gender.pt", map_location=device))

#eval models
model_age_vgg.eval()
model_gen_vgg.eval()
model_age_res.eval()
model_gen_res.eval()

#image transformation for model_age_vgg, model_gen_vgg, model_age_res
test_transforms_64 = transforms.Compose(
    [transforms.Resize((64, 64)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#image transformation for model_gen_res
test_transforms_32 = transforms.Compose(
    [transforms.Resize((32, 32)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#global frametop, frememiddle, framebottom, frameleft, frameright, fullImage
class Interface:
    def __init__(self):
        self.frametop = LabelFrame(root, text='Image recognition', width=100, height=100, borderwidth=2,
                              relief="groove")
        self.frememiddle = LabelFrame(root, text='Choose type of prediction', width=50, height=50, borderwidth=2,
                              relief="groove")
        self.framebottom = LabelFrame(root, text='Age and gender prediction', width=100, height=100, borderwidth=2,
                              relief="groove")
        self.frameleft = LabelFrame(self.frametop, text='Choose a photo', borderwidth=2, relief="groove")
        self.frameright = LabelFrame(self.frametop, text='Your photo', borderwidth=2, relief="groove")

        self.frametop.pack(side=TOP, fill=BOTH)
        self.frememiddle.pack(side=TOP, fill=BOTH)
        self.framebottom.pack(side=BOTTOM, fill=BOTH, expand=1)
        self.frameleft.pack(side=LEFT, fill=BOTH, expand=1)
        self.frameright.pack(side=RIGHT,fill=BOTH, expand=1)
    def buttons(self):
        self.takePic = Button(self.frameleft, text="Take a photo with webcam", command=make_photo)
        self.uploadPic = Button(self.frameleft, text="Upload a photo from memory", command=upload_photo)
        self.clearButton = Button(self.frameleft, text="HELP", command=show_info)
        self.quitButton = Button(self.frameleft, text="QUIT", fg="red", command=root.destroy)

        self.vgg_arh = Button(self.frememiddle, text="VGG16 Arhitecture", command=predict_image_vgg16)
        self.resnet_arh = Button(self.frememiddle, text="ResNet Arhitecture", command=predict_image_resnet)
        self.both_arh = Button(self.frememiddle, text="Both Arhitectures", command=predict_image_both)

        self.takePic.pack(side=TOP, padx=10, pady=10)
        self.uploadPic.pack(side=TOP, padx=10, pady=10)
        self.clearButton.pack(side=TOP, padx=10, pady=10)
        self.quitButton.pack(side=TOP, padx=10, pady=10)

        self.vgg_arh.pack(side=LEFT, padx=120, pady=5)
        self.vgg_arh.configure(state="disabled")
        self.resnet_arh.pack(side=LEFT, padx=120, pady=5)
        self.resnet_arh.configure(state="disabled")
        self.both_arh.pack(side=LEFT, padx=120, pady=5)
        self.both_arh.configure(state="disabled")

inter=Interface()

def activate_butons():
    global inter
    inter.vgg_arh.configure(state="active")
    inter.resnet_arh.configure(state="active")
    inter.both_arh.configure(state="active")

def make_photo():
    global fullImage, label_flag, frameright, inter
    img_counter = 0
    camera = cv2.VideoCapture(0)
    while (True):
        ret, frame = camera.read()
        try:
            cv2.imshow('frame', frame)
        except:
            pass
        k = cv2.waitKey(1)
        if k%256 == 32:
            img_name = "photo_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            camera.release()
            cv2.destroyAllWindows()
            image = Image.open(img_name)
            width, height = image.size

            procent = 250 / height
            new_height = 250
            new_width = int(procent * width)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)

            # frameright = frameright_new
            if label_flag == 1:
                fullImage.destroy()
            # upload_fullImage = Label(frameright, image=image1)
            fullImage = Label(inter.frameright, image=image)
            fullImage.image = image
            fullImage.pack(side=TOP, padx=5, pady=5)
            label_flag = 1

            activate_butons()

            processing_photo(img_name)
        elif k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break



def upload_photo():
    try:
        global fullImage, label_flag, inter, name_flag, message
        file_path = filedialog.askopenfilename()
        age = 0
        try:
            vector = file_path.split("/")
            vector = vector[::-1]
            img_name = vector[0].split(".")
            age = img_name[0]
        except:
            pass
        image = Image.open(file_path)
        width, height = image.size

        procent = 250/height
        new_height = 250
        new_width = int(procent * width)

        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        image1 = ImageTk.PhotoImage(image)
        #frameright.destroy()

        #frameright = frameright_new
        if label_flag == 1:
            fullImage.destroy()

        fullImage = Label(inter.frameright, image=image1)
        fullImage.image = image1
        fullImage.pack(side=TOP, padx=5, pady=5)
        if name_flag == 1:
            message.destroy()
        message = Label(inter.frameright, text="Image name: %s" %age, font=("Arial Bold", 10))

        message.pack(side=TOP, padx=5, pady=5)

        label_flag = 1
        name_flag = 1
        activate_butons()
        processing_photo(file_path)
    except:
        pass


def processing_photo(image):
    global image_vector, number_person, face_cascade, inter

    files = os.listdir(prediction_path)
    for file in files:
        os.remove(prediction_path + file)

    # Read the input image
    img = cv2.imread(image)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 4)
    img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = Image.fromarray(img3)
    flag = 0

    i=1
    inter.framebottom.destroy()
    inter.framebottom = LabelFrame(root, text='Age and gender prediction', width=100, height=100, borderwidth=2,
                             relief="groove")
    inter.framebottom.pack(side=BOTTOM, fill=BOTH, expand=1)
    message = Label(inter.framebottom, text="Step 1: Face detection", font=("Arial Bold", 10))

    message.pack(side=TOP, padx=5, pady=5)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        flag +=1
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        left = x
        top = y
        right = x + w
        bottom = y + h

        #crop new image
        saving_photo = img[top:bottom, left:right]
        im1 = im.crop((left, top, right, bottom))

        #new  face verification
        gray2 = cv2.cvtColor(saving_photo, cv2.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)
        flag2 = 0
        for (a, b, c, d) in faces2:
            flag2 =1


        if flag2 ==1:
            #save new image
            name_img = str(i) + ".jpg"
            dstPath = join(prediction_path, name_img)
            cv2.imwrite(dstPath, saving_photo)

            #print new image
            im1 = im1.resize((100, 100), Image.ANTIALIAS)
            image2 = ImageTk.PhotoImage(image=im1)

            person_label = LabelFrame(inter.framebottom, text='Person %d' % i,  borderwidth=2, relief="groove")

            person = Label(person_label,image=image2)
            person.image = image2
            person_label.pack(side=LEFT, padx=25, pady=20)
            person.pack(side=TOP, padx=10, pady=10)
            i +=1


    number_person = i-1

def predict_image_vgg16():
    global face_cascade, inter
    inter.framebottom.destroy()
    inter.framebottom = LabelFrame(root, text='Age and gender prediction', width=100, height=100, borderwidth=2,
                             relief="groove")
    inter.framebottom.pack(side=BOTTOM, fill=BOTH, expand=1)
    message = Label(inter.framebottom, text="VGG16 Arhitecture",font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)
    message = Label(inter.framebottom, text="Step 2: Second verification for face detection + Predictions",
                    font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)

    # second verification for face detection
    nr_pers = 0
    files = os.listdir(prediction_path)
    for img in files:
        dstPath = join(prediction_path, img)
        img = cv2.imread(dstPath)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        flag3 = 0
        for (x, y, w, h) in faces:
            flag3 = 1
            nr_pers += 1
        if flag3 == 0:
            os.remove(dstPath)
    if nr_pers == 1:
        message = Label(inter.framebottom, text="Got one person", font=("Arial Bold", 10))
    else:
        message = Label(inter.framebottom, text="Got %d person" % nr_pers, font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)

    # for every image make prediction
    i = 1
    files = os.listdir(prediction_path)
    for img in files:
        dstPath = join(prediction_path, img)
        read_img = Image.open(dstPath)
        read_img = read_img.resize((64, 64), Image.ANTIALIAS)
        image2 = ImageTk.PhotoImage(image=read_img)
        person_label = LabelFrame(inter.framebottom, text='Person %d' % i, borderwidth=2, relief="groove")

        person = Label(person_label, image=image2)
        person.image = image2
        person_label.pack(side=LEFT, padx=20, pady=20)
        person.pack(side=TOP, padx=10, pady=10)
        i +=1
        image_tensor = test_transforms_64(read_img).float()
        image_tensor = image_tensor.unsqueeze(0)

        input = Variable(image_tensor)
        input1 = input.to(device)

        est_age_vgg = model_age_vgg(input1).type(torch.float32)
        est_gen_vgg = model_gen_vgg(input1)

        predicted_gender = torch.argmax(est_gen_vgg, dim=1)
        y = est_age_vgg * 100
        if nr_pers == 1:
            if predicted_gender == 1:
                gen_vgg_label = Label(person_label, text="Gender : female", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
            else:
                gen_vgg_label = Label(person_label, text="Gender : male", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
        else:
            if predicted_gender == 0:
                gen_vgg_label = Label(person_label, text="Gender : female", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
            else:
                gen_vgg_label = Label(person_label, text="Gender : male", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
        age_vgg_label = Label(person_label, text="Estimated Age : %.1f" % y, font=("Arial Bold", 10))
        age_vgg_label.pack(side=BOTTOM, padx=5, pady=5)


def predict_image_resnet():
    global face_cascade, inter
    inter.framebottom.destroy()
    inter.framebottom = LabelFrame(root, text='Age and gender prediction', width=100, height=100, borderwidth=2, relief="groove")
    inter.framebottom.pack(side=BOTTOM, fill=BOTH, expand=1)
    message = Label(inter.framebottom, text="ResNet50 Arhitecture", font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)
    message = Label(inter.framebottom, text= "Step 2: Second verification for face detection + Predictions", font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)

    # second verification for face detection
    nr_pers = 0
    files = os.listdir(prediction_path)
    for img in files:
        dstPath = join(prediction_path, img)
        img = cv2.imread(dstPath)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        flag3 = 0
        for (x, y, w, h) in faces:
            flag3 = 1
            nr_pers += 1
        if flag3 == 0:
            os.remove(dstPath)
    if nr_pers ==1:
        message = Label(inter.framebottom, text="Got one person",font=("Arial Bold", 10))
    else:
        message = Label(inter.framebottom, text="Got %d person" %nr_pers, font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)

    #for every image make prediction
    i = 1
    files = os.listdir(prediction_path)
    for img in files:
        dstPath = join(prediction_path, img)

        read_img = Image.open(dstPath)
        read_img = read_img.resize((64, 64), Image.ANTIALIAS)
        image2 = ImageTk.PhotoImage(image=read_img)
        person_label = LabelFrame(inter.framebottom, text='Person %d' % i, borderwidth=2, relief="groove")

        person = Label(person_label, image=image2)
        person.image = image2
        person_label.pack(side=LEFT, padx=20, pady=20)
        person.pack(side=TOP, padx=10, pady=10)
        i +=1
        image_tensor_32p = test_transforms_32(read_img).float()
        image_tensor_32p = image_tensor_32p.unsqueeze(0)

        input_32 = Variable(image_tensor_32p)
        input2 = input_32.to(device)

        est_age_resnet = model_age_res(input2).type(torch.float32)
        est_gen_resnet = model_gen_res(input2)

        predicted_gender_res = torch.argmax(est_gen_resnet, dim=1)
        age_res = est_age_resnet * 100
        if nr_pers == 1:
            if predicted_gender_res == 1:
                gen_vgg_label = Label(person_label, text="Gender : female", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
            else:
                gen_vgg_label = Label(person_label, text="Gender : male", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
        else:
            if predicted_gender_res == 0:
                gen_vgg_label = Label(person_label, text="Gender : female", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
            else:
                gen_vgg_label = Label(person_label, text="Gender : male", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)

        age_res_label = Label(person_label, text="Estimated Age : %.1f" % age_res, font=("Arial Bold", 10))
        age_res_label.pack(side=BOTTOM, padx=5, pady=5)


def predict_image_both():
    global face_cascade, inter
    inter.framebottom.destroy()
    inter.framebottom = LabelFrame(root, text='Age and gender prediction', width=100, height=100, borderwidth=2,
                             relief="groove")
    inter.framebottom.pack(side=BOTTOM, fill=BOTH, expand=1)
    message = Label(inter.framebottom, text="Predictions for VGG16 and ResNet50", font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)
    message = Label(inter.framebottom, text="Step 2: Second verification for face detection + Predictions",
                    font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)

    # second verification for face detection
    nr_pers = 0
    files = os.listdir(prediction_path)
    for img in files:
        dstPath = join(prediction_path, img)
        img = cv2.imread(dstPath)
        # Convert into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        flag3 = 0
        for (x, y, w, h) in faces:
            flag3 = 1
            nr_pers += 1
        if flag3 == 0:
            os.remove(dstPath)
    if nr_pers == 1:
        message = Label(inter.framebottom, text="Got one person", font=("Arial Bold", 10))
    else:
        message = Label(inter.framebottom, text="Got %d person" % nr_pers, font=("Arial Bold", 10))
    message.pack(side=TOP, padx=5, pady=5)

    # for every image make prediction
    i = 1
    files = os.listdir(prediction_path)
    for img in files:
        dstPath = join(prediction_path, img)

        read_img = Image.open(dstPath)
        read_img = read_img.resize((64, 64), Image.ANTIALIAS)
        image2 = ImageTk.PhotoImage(image=read_img)

        image_tensor_64p = test_transforms_64(read_img).float()
        image_tensor_32p = test_transforms_32(read_img).float()

        image_tensor_64p = image_tensor_64p.unsqueeze(0)
        image_tensor_32p = image_tensor_32p.unsqueeze(0)

        input_64 = Variable(image_tensor_64p)
        input_32 = Variable(image_tensor_32p)

        input1 = input_64.to(device)
        input2 = input_32.to(device)

        est_age_vgg = model_age_vgg(input1).type(torch.float32)
        est_gen_vgg = model_gen_vgg(input1)
        predicted_gender_vgg = torch.argmax(est_gen_vgg, dim=1)

        est_age_resnet = model_age_res(input2).type(torch.float32)
        est_gen_resnet = model_gen_res(input2)
        predicted_gender_res = torch.argmax(est_gen_resnet, dim=1)

        age_vgg = est_age_vgg * 100
        age_res = est_age_resnet * 100

        if nr_pers == 1:
            person_label1 = LabelFrame(inter.framebottom, text='VGG16 Arhitecture', borderwidth=2, relief="groove")
            person_label2 = LabelFrame(inter.framebottom, text='ResNet50 Arhitecture' , borderwidth=2, relief="groove")

            person = Label(person_label1, image=image2)
            person.image = image2
            person_label1.pack(side=LEFT, padx=20, pady=20)
            person.pack(side=TOP, padx=10, pady=10)

            person = Label(person_label2, image=image2)
            person.image = image2
            person_label2.pack(side=LEFT, padx=20, pady=20)
            person.pack(side=TOP, padx=10, pady=10)


            if predicted_gender_vgg == 1:
                gen_vgg_label = Label(person_label1, text="Gender : female", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
            else:
                gen_vgg_label = Label(person_label1, text="Gender : male", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=5)
            age_vgg_label = Label(person_label1, text="Estimated Age : %.1f" % age_vgg, font=("Arial Bold", 10))
            age_vgg_label.pack(side=BOTTOM, padx=5, pady=5)

            if predicted_gender_res == 1:
                gen_res_label = Label(person_label2, text="Gender : female", font=("Arial Bold", 10))
                gen_res_label.pack(side=BOTTOM, padx=5, pady=5)
            else:
                gen_res_label = Label(person_label2, text="Gender : male", font=("Arial Bold", 10))
                gen_res_label.pack(side=BOTTOM, padx=5, pady=5)
            age_res_label = Label(person_label2, text="Estimated Age : %.1f" % age_res, font=("Arial Bold", 10))
            age_res_label.pack(side=BOTTOM, padx=5, pady=5)
        else:
            person_label = LabelFrame(inter.framebottom, text='Person %d' % i, borderwidth=2, relief="groove")
            person = Label(person_label, image=image2)
            person.image = image2
            person_label.pack(side=LEFT, padx=20, pady=10)
            person.pack(side=TOP, padx=5, pady=5)

            vgg_label = LabelFrame(person_label, text='VGG16 Arhitecture' , borderwidth=2, relief="groove")
            res_label = LabelFrame(person_label, text='ResNet50 Arhitecture', borderwidth=2, relief="groove")

            vgg_label.pack(side=TOP, padx=5, pady=3)
            res_label.pack(side=TOP, padx=5, pady=3)



            if predicted_gender_vgg == 0:
                gen_vgg_label = Label(vgg_label, text="Gender : female", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=3)
            else:
                gen_vgg_label = Label(vgg_label, text="Gender : male", font=("Arial Bold", 10))
                gen_vgg_label.pack(side=BOTTOM, padx=5, pady=3)
            age_vgg_label = Label(vgg_label, text="Estimated Age : %.1f" % age_vgg, font=("Arial Bold", 10))
            age_vgg_label.pack(side=BOTTOM, padx=5, pady=3)

            if predicted_gender_res == 0:
                gen_res_label = Label(res_label, text="Gender : female", font=("Arial Bold", 10))
                gen_res_label.pack(side=BOTTOM, padx=5, pady=3)
            else:
                gen_res_label = Label(res_label, text="Gender : male", font=("Arial Bold", 10))
                gen_res_label.pack(side=BOTTOM, padx=5, pady=3)
            age_res_label = Label(res_label, text="Estimated Age : %.1f" % age_res, font=("Arial Bold", 10))
            age_res_label.pack(side=BOTTOM, padx=5, pady=3)
            i +=1

def show_info():
    messagebox.showinfo(title='Age and gender prediction', message='Step 1: Choose a method to add an image. '
                                                                   'If you take a photo with webcam you have to press the "SPACE" button '
                                                                   'to take the picture or "ESC" to exit the camera window. '
                                                                   'After uploading the input image, facial extraction will be performed automatically.   '
                                                                   'Step 2: Choose a prediction method by pressing one of the three buttons. '
                                                                   'You can change the prediction method at any time. '
                                                                   'If you want predictions for another image, repeat the first two steps.')



if __name__ == "__main__":
    inter.buttons()
    root.mainloop()