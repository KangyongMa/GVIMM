# Use Azure’s Python base image
FROM mcr.microsoft.com/azure-app-service/python:3.9_20250213.3.tuxprod

# Install X11 libs needed by RDKit drawing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libcairo2-dev libxcb1 libxrender1 libxext6 libice6 libsm6 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/site/wwwroot

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV PORT 8000
CMD ["bash", "startup.sh"]