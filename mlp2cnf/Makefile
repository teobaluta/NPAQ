PBLIB=pblib
CFLAGS=-g #-Wall
TARGET=bin/mlp2cnf
INCLUDE_DIR=include
SRC_DIR=src

#SOURCES=$(shell find $(SRC_DIR) -type f -name *.cpp)

all: dep
	$(CXX) -std=c++11 $(CFLAGS) -I $(INCLUDE_DIR) -o bin/bnn2cnf src/utils.cpp src/bnn.cpp src/constraint_to_cnf.cpp src/adv_bnn.cpp src/bnn_prop.cpp src/bnn2cnf.cpp src/bnn_check.cpp -I$(PBLIB) $(PBLIB)/lib/libpblib.a


dep:
	@echo "Checking dep pblib"
	@if [ ! -f $(PBLIB) ]; then \
		mkdir $(PBLIB); tar -xvf pblib_npaq.tar.gz; \
	fi
	@if [ ! -f $(PBLIB)/lib/libpblib.a ]; then \
		echo "Building pblib library"; \
		cd $(PBLIB); \
		cmake . ; \
		make setup; \
		cd - ;\
	fi
	@if [ ! -f $(TARGET) ]; then echo "Creating bin/ directory" ; mkdir -p bin/ ; fi

clean:
	rm -rf bin/
