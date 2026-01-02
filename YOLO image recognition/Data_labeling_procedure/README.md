# How to automate label with gliner and docker
- download ngrok, docker desktop
- setup auth key for docker 

1) clone 
```bash
https://github.com/HumanSignal/label-studio-ml-backend
```

2) open docker, open command prompt, install label-studio and open it
``` bash
label-studio
```

3) change the directory to the gliner folder
```bash
cd F:\github\label-studio-ml-backend\label_studio_ml\examples\gliner
```
```bash
docker compose up -d
```

4) Verify Local Access
```bash
curl http://localhost:9090
```

5) 
```bash
ngrok http 9090 
```

