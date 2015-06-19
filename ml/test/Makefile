###############################################################################
#                              COMMON SETTINGS                                #
###############################################################################
EXEC_NAME = test.x
CUSTOM_LD_FLAGS = -lgtest_main -pthread -lnlopt -lm
CUSTOM_CC_FLAGS = 
ifdef LARGE
  CUSTOM_CC_FLAGS += -DLARGE_NN
endif
ifdef VERBOSE
  CUSTOM_CC_FLAGS += -DVERBOSE
endif

OBJS = \
../../math/test/math.test.o \
../linear_regression.o \
../gradient_descent.o \
../neural_network.o \
ml_gradient_descent.test.o \
ml_linear_regression.test.o \
ml_neural_network.test.o \
ml_vec_mat_math.test.o \

###############################################################################
#                            INCLUDE AND LIB DIRS                             #
###############################################################################
BOOST_DIR = /usr/local/include/boost
THRIFT_DIR = /usr/local/include
XEN_DIR = ${XEN_ROOT}
GTEST_DIR = ${GTEST_ROOT}

###############################################################################
#                                    FLAGS                                    #
###############################################################################

SHELL = bash
CXX = g++
CC = gcc
OPT_LEVEL = -O3
ifdef DEBUG
  OPT_LEVEL = -g
endif
CCFLAGS = -c ${OPT_LEVEL} -I${XEN_DIR} -I${THRIFT_DIR} -I${BOOST_DIR} -I. \
          -I${GTEST_DIR}/include ${CUSTOM_CC_FLAGS}
CXXFLAGS = $(CCFLAGS) -std=c++11
LDFLAGS = -L/usr/local/lib -L${GTEST_DIR}/lib

###############################################################################
#                                    RULES                                    #
###############################################################################

default: $(EXEC_NAME)

$(EXEC_NAME): $(OBJS:.o=.d) $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) $(CUSTOM_LD_FLAGS) -o $@

-include $(OBJS:.o=.d)

%.d: %.cpp
	$(SHELL) -ec '$(CXX) -M $(CXXFLAGS) $< | sed "s|$*.o|& $@|g" > $@'

%.d: %.c
	$(SHELL) -ec '$(CC) -M $(CCFLAGS) $< | sed "s|$*.o|& $@|g" > $@'

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $< -o $@

clean:
	$(RM) -rf $(OBJS) $(OBJS:.o=.d) $(EXEC_NAME)

