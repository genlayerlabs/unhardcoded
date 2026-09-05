"""AWS credentials from the request scope, never the operator chain for BYO."""


def client(service, region, env_get, **kwargs):
    import boto3
    if env_get('SAAS_TENANT_SCOPE'):
        key, secret = env_get('AWS_ACCESS_KEY_ID'), env_get('AWS_SECRET_ACCESS_KEY')
        if not key or not secret:
            raise ValueError('Workspace AWS credentials are missing')
        # Explicit credentials disable profile, container-role and IMDS fallback.
        kwargs.update(aws_access_key_id=key, aws_secret_access_key=secret,
                      aws_session_token=env_get('AWS_SESSION_TOKEN') or None)
    return boto3.client(service, region_name=region, **kwargs)
