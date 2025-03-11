BLIS_INSTALL_DIR := $(HOME)/blis

-include path.mk

INC   := include/

BLIS_LIB  := $(BLIS_INSTALL_DIR)/lib/libblis.a
BLIS_INC  := $(BLIS_INSTALL_DIR)/include/blis


CC         := gcc
LINKER     := $(CC) 
CFLAGS     := -O3 -I$(BLIS_INC) -I$(INC) -m64 -mavx2 -mfma \
              -mfpmath=sse -std=c99 -march=core-avx2 \
              -D_POSIX_C_SOURCE=200112L -fopenmp -funroll-loops
CDEBUG     := -g
SRC_PATH   := .
LDFLAGS	   := -lm -lpthread


SRC=$(shell find ${SRC_PATH} -type f -iname '*.c')
OBJS=$(patsubst %.c,%.o, ${SRC})


#$(info print source $(SRC))
#$(info print target $(OBJS))


driver: $(OBJS)
	$(LINKER) $(OBJS) $(BLIS_LIB) -o driver.x $(LDFLAGS) 


%.o: %.c
	$(CC) $(CDEBUG) $(CFLAGS) -c $< -o $@


clean:
	rm -f ${OBJS} *.x
