FROM nvcr.io/nvidia/tritonserver:23.10-py3

# I only install cpu version of torch since GPU is not required for inference with sentence-transformers
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    sentence-transformers \
    faiss-cpu \
    numpy

WORKDIR /build
COPY build_repo.py .
COPY model.py .

RUN mkdir /models
# python3 must be used to match tritonserver's python version
RUN python3 build_repo.py

CMD ["tritonserver", "--model-repository=/models"]