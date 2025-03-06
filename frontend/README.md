Steps to deploy:

docker build . -t queriegeniefrontend --build-arg env=dev
docker tag queriegeniefrontend sdplzmgmtdcacr01.azurecr.io/query-genie-frontend:latest
docker push sdplzmgmtdcacr01.azurecr.io/query-genie-frontend:latest

To run locally:

docker run -p 3000:3000 queriegeniefrontend

