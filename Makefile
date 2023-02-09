TAG=0.1
HOST=in2p3.fr
PLATFORM=gitlab
REGISTRY=$(PLATFORM)-registry.$(HOST)
REPONAME=machine_learning_for_climate_and_energy
IMAGE=$(REGISTRY)/energy4climate/public/education/$(REPONAME):$(TAG)

HOSTNAME=$(PLATFORM).$(HOST)
BASEDIR=energy4climate/public/education
BASEURL=https://$(HOSTNAME)/$(BASEDIR)
REPO=$(BASEURL)/$(REPONAME).git

build:
	repo2docker --debug --no-run --user-id 1000 --user-name jovyan --image-name $(IMAGE) $(REPO)
debug:
	repo2docker --debug --no-build --no-run --user-id 1000 --user-name jovyan --image-name $(IMAGE) $(REPO)
login:
	docker login $(HOSTNAME)
push:
	docker push $(IMAGE)
lab:
	docker run --name lab -it --rm -p 8888:8888 $(IMAGE) jupyter notebook --NotebookApp.default_url=/lab/ --ip=0.0.0.0 --port=8888
peek:
	docker run -it --rm -p 8888:8888 $(IMAGE) bash
