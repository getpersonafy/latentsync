## Latentsync 1.5

### 1. Build the image

```
docker buildx build --load --platform linux/amd64 -t latentsync-dev .
```

### 2. Tag the image

```
docker tag latentsync-dev:latest <repository>/latentsync-dev:<date>
```

<b>Note</b>: If multiple deployments on the same date, the format will be: `latentsync-dev:<date>-<num>`.
<br>
<b>Suggested format</b>: 7.23.2025 and 7.23.2025-2 (if built and deployed on the same date)

### 3. Push the image to ECR

```
docker push <repository>/latentsync-dev:<date>
```
