from deepface import DeepFace
obj = DeepFace.verify("./uploaded_image.jpg", "./name_122.jpg"
          , model_name = 'ArcFace', detector_backend = 'retinaface')
print(obj["verified"])

