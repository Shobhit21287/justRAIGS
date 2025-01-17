FROM --platform=linux/amd64  pytorch/pytorch

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN adduser --system --group user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app
COPY --chown=algorithm:algorithm model.pth /opt/algorithm/model.pth
COPY --chown=algorithm:algorithm model_one_hot.pth /opt/algorithm/model_one_hot.pth

RUN python -m pip install \
    --no-color \
    --requirement requirements.txt

COPY --chown=user:user helper.py /opt/app
COPY --chown=user:user inference.py /opt/app

ENTRYPOINT ["python", "inference.py"]
