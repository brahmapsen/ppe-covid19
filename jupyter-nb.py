#USE PPE Detector for Laboratory Safety

#Initializing a model using AWS SageMaker Python API
import sagemaker as sage
from sagemaker import get_execution_role

sess = sage.Session()
role = get_execution_role()

# from utils import get_model_package_arn
# model_package_arn = get_model_package_arn(sess.boto_region_name)

model_package_arn = 'arn:aws:sagemaker:us-east-2:057799348421:model-package/vitechlab-ppe-lab-model-v1-1-480cb3fc5bd97e9bf1e213cca498bbc4'

model = sage.ModelPackage(
    role=role,
    model_package_arn=model_package_arn)

#Generating Prediction using RealTimePredictor
#Creating end point
#endpoint_name='ppe-lab-model-example-endpoint'
endpoint_name='ppe-covid19-endpoint'
model.deploy(initial_instance_count=1, instance_type='ml.m4.2xlarge', endpoint_name=endpoint_name)

#Generating Prediction
predictor = sage.predictor.RealTimePredictor(
    endpoint_name,
    sagemaker_session=sess, 
    content_type="image/jpeg"
)

#------------------------PREDICTION WITH IMAGES WITH MASK
file_name = './pers1-mask.jpeg'
with open(file_name, 'rb') as image:
    f = image.read()
    image_bytes = bytearray(f)
prediction_result = predictor.predict(image_bytes).decode("utf-8")
import json
result = json.loads(prediction_result)
result

#OUTPUT
# [{'name': 'person',
#   'percentage_probability': 99.87660646438599,
#   'box_points': [36, 17, 218, 186],
#   'classes': {'no_coat': 24.943213164806366,
#    'no_glasses': 99.97839331626892,
#    'no_glove': 0.0014959524378355127,
#    'no_mask': 0.030190424877218902}}]


file_name = './pers2-mask.jpeg'
with open(file_name, 'rb') as image:
    f = image.read()
    image_bytes = bytearray(f)
prediction_result = predictor.predict(image_bytes).decode("utf-8")
import json
result = json.loads(prediction_result)
result

#OUTPUT
# [{'name': 'person',
#   'percentage_probability': 96.63254618644714,
#   'box_points': [123, 1, 258, 192],
#   'classes': {'no_coat': 1.5348253771662712,
#    'no_glasses': 27.06289291381836,
#    'no_glove': 0.036967472988180816,
#    'no_mask': 95.9999680519104}}]

file_name = './pers3-mask.jpeg'
with open(file_name, 'rb') as image:
    f = image.read()
    image_bytes = bytearray(f)
prediction_result = predictor.predict(image_bytes).decode("utf-8")
import json
result = json.loads(prediction_result)
result
#OUTPUT
# [{'name': 'person',
#   'percentage_probability': 97.53351211547852,
#   'box_points': [117, 6, 254, 180],
#   'classes': {'no_coat': 96.0574209690094,
#    'no_glasses': 98.77341389656067,
#    'no_glove': 99.76134300231934,
#    'no_mask': 24.380697309970856}}]


file_name = './pers4-mask.jpeg'
with open(file_name, 'rb') as image:
    f = image.read()
    image_bytes = bytearray(f)
prediction_result = predictor.predict(image_bytes).decode("utf-8")
import json
result = json.loads(prediction_result)
result

#OUTPUT
# [{'name': 'person',
#   'percentage_probability': 78.7774920463562,
#   'box_points': [119, 2, 255, 165],
#   'classes': {'no_coat': 0.1960463123396039,
#    'no_glasses': 95.38192749023438,
#    'no_glove': 0.004500397699302994,
#    'no_mask': 0.01127810319303535}}]





#PREDICT WITH NO MASK

file_name = './pers1-no-mask.jpeg'
with open(file_name, 'rb') as image:
    f = image.read()
    image_bytes = bytearray(f)
prediction_result = predictor.predict(image_bytes).decode("utf-8")
import json
result = json.loads(prediction_result)
result

#OUTPUT
# [{'name': 'person',
#   'percentage_probability': 99.07587766647339,
#   'box_points': [0, 21, 111, 168],
#   'classes': {'no_coat': 2.4690108373761177,
#    'no_glasses': 98.43910932540894,
#    'no_glove': 0.00014549445950251538,
#    'no_mask': 99.97814297676086}}]

​
file_name = './pers2-no-mask.jpeg'
with open(file_name, 'rb') as image:
    f = image.read()
    image_bytes = bytearray(f)
prediction_result = predictor.predict(image_bytes).decode("utf-8")
import json
result = json.loads(prediction_result)
result


#OUTPUT
# [{'name': 'person',
#   'percentage_probability': 97.67805933952332,
#   'box_points': [5, 0, 183, 274],
#   'classes': {'no_coat': 99.99905824661255,
#    'no_glasses': 99.99995231628418,
#    'no_glove': 23.195715248584747,
#    'no_mask': 99.9998927116394}}]



from utils import visualize_detection

visualize_detection(file_name, result)

#Delete end point
sess.delete_endpoint(predictor.endpoint)

