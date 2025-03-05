Steps to deploy:

docker build . -t queriegeniebe --build-arg env=dev
docker tag queriegeniebe sdplzmgmtdcacr01.azurecr.io/query-genie-backend:latest
docker push sdplzmgmtdcacr01.azurecr.io/query-genie-backend:latest
