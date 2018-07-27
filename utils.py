import tensorflow as tf
from data_loader import convert_back


def log_b(x, base=10):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
  return numerator / denominator


def psnr(hr_image, sr_image):
  """
  Peak signal-to-noise ratio
  """
  with tf.name_scope('psnr'):
    hr_image = convert_back(hr_image, LR=False)
    sr_image = convert_back(sr_image, LR=False)
    mse = tf.reduce_mean(tf.square(hr_image - sr_image))
    psnr = 20 * log_b(255. / tf.sqrt(mse), base=10)
  return psnr

def crop_images(filepath, size=24):
  cnt = 0

  for pic in range(1, (numPics + 1)):
    img = cv2.imread('input/' + str(pic) + '.jpg')
    height = img.shape[0]
    width = img.shape[1]
    size = height * width

    if size > (500 ^ 2):
      r = 500.0 / img.shape[1]
      dim = (500, int(img.shape[0] * r))
      img2 = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
      img = img2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
      eyesn = 0
      imgCrop = img[y:y + h, x:x + w]
      # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y + h, x:x + w]
      roi_color = img[y:y + h, x:x + w]

      eyes = eye_cascade.detectMultiScale(roi_gray)
      for (ex, ey, ew, eh) in eyes:
        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        eyesn = eyesn + 1
      if eyesn >= 2:
        #### increase the counter and save
        cnt += 1
        cv2.imwrite("output/crop{}_{}.jpg".format(pic, cnt), imgCrop)

        # cv2.imshow('img',imgCrop)
        print("Image" + str(pic) + " has been processed and cropped")

    k = cv2.waitKey(100) & 0xff
    if k == 27:
      break

  # cap.release()
  print("All images have been processed!!!")
  cv2.destroyAllWindows()
  cv2.destroyAllWindows()