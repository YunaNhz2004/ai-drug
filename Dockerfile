FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -c conda-forge \
    rdkit \
    python=3.10 \
    pillow \
    cairo \
    -y && \
    conda clean -afy

RUN pip install --no-cache-dir \
    numpy==2.1.3 \
    pandas==2.2.3

RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric==2.4.0 \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user . /app

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]