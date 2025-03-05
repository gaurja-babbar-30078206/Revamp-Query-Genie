Steps to deploy:

docker build . -t queriegeniefe --build-arg env=dev
docker tag queriegeniefe sdplzmgmtdcacr01.azurecr.io/query-genie-fe:latest
docker push sdplzmgmtdcacr01.azurecr.io/query-genie-fe:latest



