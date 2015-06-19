###############################################################################
#                              COMMON SETTINGS                                #
###############################################################################
EXEC_NAME = test.x
CUSTOM_FLAGS = -lgtest_main -pthread

OBJS = \
time.test.o

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
CCFLAGS = -c -g -I${XEN_DIR} -I${THRIFT_DIR} -I${BOOST_DIR} -I. \
          -I${GTEST_DIR}/include
CXXFLAGS = $(CCFLAGS) -std=c++11
LDFLAGS = -L/usr/local/lib -L${GTEST_DIR}/lib

###############################################################################
#                                    RULES                                    #
###############################################################################

default: $(EXEC_NAME)

$(EXEC_NAME): $(OBJS:.o=.d) $(OBJS)
	$(CXX) $(LDFLAGS) $(CUSTOM_FLAGS) $(OBJS) -o $@

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

