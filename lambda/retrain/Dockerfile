FROM public.ecr.aws/lambda/python:3.11

COPY handler.py ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    imbalanced-learn==0.12.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    joblib==1.4.2

CMD ["handler.lambda_handler"]