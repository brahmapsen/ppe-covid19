#USE PPE Detector for Laboratory Safety

#Initializing a model using AWS SageMaker Python API
import sagemaker as sage
from sagemaker import get_execution_role

sess = sage.Session()
role = get_execution_role()

from utils import get_model_package_arn

model_package_arn = get_model_package_arn(sess.boto_region_name)

model = sage.ModelPackage(
    role=role,
    model_package_arn=model_package_arn)

#Generating Prediction using RealTimePredictor
#Creating end point
endpoint_name='ppe-lab-model-example-endpoint'

model.deploy(initial_instance_count=1, instance_type='ml.m4.2xlarge', endpoint_name=endpoint_name)

#Generating Prediction
predictor = sage.predictor.RealTimePredictor(
    endpoint_name,
    sagemaker_session=sess, 
    content_type="image/jpeg"
)

#S3 bucket name bcuser-bps-ml-sample-s3
#uploading input data to S3
input_data_folder = 'sample_data/demo_input'
#s3_data_folder = 'ppe_lab_model_batch_testing/input'
s3_data_folder = 'bcuser-bps-ml-sample-s3'

input_data_location = sess.upload_data(input_data_folder, key_prefix=s3_data_folder)
output_data_location = input_data_location.replace('input', 'output')

#start batch job
transformer = model.transformer(instance_count=1,
                               instance_type='ml.c4.2xlarge',
                               output_path=output_data_location,
                               strategy='SingleRecord',
                               assemble_with=None,
                               accept='image/jpeg')

transformer.transform(input_data_location, 
    content_type='image/jpeg',
    compression_type=None,
    split_type=None,
    join_source=None)

transformer.wait()

#download and display result
output_file_name = 'image1.jpg.out'
dest_path = 'sample_data/demo_raw_output/' + output_file_name
#s3_path = 'ppe_lab_model_batch_testing/output/' + output_file_name
s3_path = 'bcuser-bps-ml-sample-s3/' + output_file_name

bucket = sess.boto_session.resource('s3').Bucket(sess.default_bucket())
bucket.download_file(s3_path, dest_path)

with open(dest_path, 'r') as file:
    print(file.read())



#Delete end point
sess.delete_endpoint(predictor.endpoint)

