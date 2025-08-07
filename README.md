In order to run this with Docker, we need first to build the Docker image.
From the root of the repository, run:
```sh
docker build -t segment_spectral .
```

Once the image is built, you can run the application with:
```sh
docker compose up -d
```