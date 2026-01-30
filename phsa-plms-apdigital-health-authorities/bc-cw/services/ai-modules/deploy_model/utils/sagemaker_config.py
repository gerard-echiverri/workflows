from botocore.exceptions import ClientError

def create_endpoint_config(sagemaker_client, config_name, model_name, instance_type):
    """
    Create a SageMaker endpoint config with the specified parameters.
    """
    try:
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': 1
            }]
        )
        print(f"Created endpoint config: {config_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in str(e):
            print(f"Endpoint config {config_name} already exists.")
        else:
            raise
