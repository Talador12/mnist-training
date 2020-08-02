## Makefile as a quick runbook
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## -------------

## var - $(venv)

VENV=venv/bin/

## var - $(OSFLAG)
OSFLAG 				:=
ifeq ($(OS),Windows_NT)
	OSFLAG += -D WIN32
	ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		OSFLAG += -D AMD64
	endif
	ifeq ($(PROCESSOR_ARCHITECTURE),x86)
		OSFLAG += -D IA32
	endif
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OSFLAG += -D LINUX
		source += .
	endif
	ifeq ($(UNAME_S),Darwin)
		OSFLAG += -D OSX
	endif
		UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_P),x86_64)
		OSFLAG += -D AMD64
	endif
		ifneq ($(filter %86,$(UNAME_P)),)
	OSFLAG += -D IA32
		endif
	ifneq ($(filter arm%,$(UNAME_P)),)
		OSFLAG += -D ARM
	endif
endif

## -------------

## make pre-install - creates virtual environment and upgrades pip
pre-install:
	pip3 install -i https://pypi.org/simple --upgrade virtualenv==20.0.28
	virtualenv --python=/usr/bin/python3.8 venv
	$(VENV)pip3 install -i https://pypi.org/simple --upgrade pip

## make install - runs pre-install in a virtual environment and installs standard dependencies
install:
	make pre-install
	$(VENV)pip3 install -i https://pypi.org/simple --upgrade -r requirements.txt

## make install-dev - runs pre-install in a virtual environment and installs standard + dev dependencies
install-dev:
	make pre-install
	$(VENV)pip3 install -i https://pypi.org/simple --upgrade -r requirements.txt

## make install-test - runs pre-install in a virtual environment and installs standard + test dependencies
install-test:
	make pre-install
	$(VENV)pip3 install -i https://pypi.org/simple --upgrade -r requirements/requirements-test.txt

## make clean - cleans up the repo
clean:
	rm -rf venv
	rm pyvenv.cfg || true
	find -iname "*.pyc" -delete

## make tfnn - trains a tensorflow neural net on the mnist dataset
tfnn:
	$(VENV)python3 training/tfnn.py
