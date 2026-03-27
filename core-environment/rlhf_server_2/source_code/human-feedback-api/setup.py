from setuptools import setup

setup(name='human_feedback_api',
    version='0.0.1',
    install_requires=[
        'django==1.11',
        'dj_database_url',
        'djangorestframework==3.9.0',
        'gunicorn',
        'whitenoise',
        'ipython',
        'psycopg2-binary==2.8.6',
    ]
)
